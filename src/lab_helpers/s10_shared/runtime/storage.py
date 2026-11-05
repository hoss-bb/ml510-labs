"""Small durable state store; each operation owns a short transaction."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import uuid


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, checksum TEXT NOT NULL, status TEXT NOT NULL,
                    total INTEGER DEFAULT 0, accepted INTEGER DEFAULT 0, rejected INTEGER DEFAULT 0,
                    seconds REAL DEFAULT 0, error TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS listings (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, record TEXT NOT NULL, prediction TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rejections (
                    batch_id TEXT NOT NULL, row_number INTEGER NOT NULL, listing_id TEXT,
                    reason TEXT NOT NULL, PRIMARY KEY(batch_id, row_number));
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, listing_id TEXT NOT NULL,
                    status TEXT NOT NULL, snapshot TEXT NOT NULL, stages TEXT NOT NULL DEFAULT '{}',
                    result TEXT, error TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS jobs_listing ON jobs(listing_id);
            """)

    def query(self, sql, values=()):
        with self.connection() as db:
            return [dict(row) for row in db.execute(sql, values).fetchall()]

    def check_ready(self):
        self.query("SELECT 1 FROM batches LIMIT 1")

    def batches(self, *, offset=0, limit=250):
        return self.query("SELECT * FROM batches ORDER BY id LIMIT ? OFFSET ?", (limit, offset))

    def batch(self, batch_id):
        rows = self.query("SELECT checksum,status FROM batches WHERE id=?", (batch_id,))
        return rows[0] if rows else None

    def rejections(self, batch_id, *, offset=0, limit=250):
        return self.query("SELECT * FROM rejections WHERE batch_id=? ORDER BY row_number LIMIT ? OFFSET ?",
                          (batch_id, limit, offset))

    def listings(self, *, offset=0, limit=50):
        rows = self.query("SELECT * FROM listings ORDER BY batch_id,id LIMIT ? OFFSET ?", (limit, offset))
        return [{"batch_id": row["batch_id"], "listing": json.loads(row["record"]),
                 "prediction": json.loads(row["prediction"])} for row in rows]

    def listing_reviews(self, listing_id, *, offset=0, limit=250):
        return self.query("SELECT id,status,error,created FROM jobs WHERE listing_id=? ORDER BY rowid DESC LIMIT ? OFFSET ?",
                          (listing_id, limit, offset))

    def begin_batch(self, batch_id, checksum):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if row and row["checksum"] != checksum:
                raise Conflict("Existing batch ID has changed contents.")
            if row and row["status"] == "completed":
                return False
            db.execute("INSERT INTO batches(id,checksum,status) VALUES(?,?,'running') "
                       "ON CONFLICT(id) DO UPDATE SET status='running',error=NULL,updated=CURRENT_TIMESTAMP",
                       (batch_id, checksum))
            return True

    def finish_batch(self, batch_id, accepted, rejected, seconds):
        # Publish all results and completion together. A crash before this transaction is retryable.
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM rejections WHERE batch_id=?", (batch_id,))
            stored = 0
            rejected = list(rejected)
            for row_number, listing, prediction in accepted:
                if db.execute("SELECT 1 FROM listings WHERE id=?", (listing.listing_id,)).fetchone():
                    rejected.append((row_number, listing.listing_id, "duplicate_listing_id"))
                    continue
                db.execute("INSERT INTO listings VALUES(?,?,?,?)", (listing.listing_id, batch_id,
                           listing.model_dump_json(), prediction.model_dump_json()))
                stored += 1
            for row_number, listing_id, reason in rejected:
                db.execute("INSERT INTO rejections VALUES(?,?,?,?)", (batch_id, row_number, listing_id, reason))
            total = stored + len(rejected)
            db.execute("UPDATE batches SET status='completed',total=?,accepted=?,rejected=?,seconds=?,"
                       "error=NULL,updated=CURRENT_TIMESTAMP WHERE id=?",
                       (total, stored, len(rejected), seconds, batch_id))
        return {"total": total, "accepted": stored, "rejected": len(rejected)}

    def fail_batch(self, batch_id, error):
        with self.connection() as db:
            db.execute("UPDATE batches SET status='failed',error=?,updated=CURRENT_TIMESTAMP WHERE id=?",
                       (error, batch_id))

    def listing(self, listing_id):
        rows = self.query("SELECT * FROM listings WHERE id=?", (listing_id,))
        if not rows:
            raise KeyError(listing_id)
        row = rows[0]
        return {"batch_id": row["batch_id"], "listing": json.loads(row["record"]),
                "prediction": json.loads(row["prediction"])}

    def workspace(self, *, offset=0, limit=50, neighbourhood=None, room_type=None,
                  batch_id=None, review_status=None, search=""):
        """Filter the full inventory, then paginate; review status means latest attempt."""
        inventory = """WITH inventory AS (
            SELECT l.*, COALESCE(j.status, 'not_requested') AS review_status
            FROM listings l LEFT JOIN jobs j ON j.rowid = (
                SELECT rowid FROM jobs WHERE listing_id=l.id ORDER BY rowid DESC LIMIT 1)) """
        conditions, values = [], []
        fields = {"neighbourhood": neighbourhood, "room_type": room_type}
        for field, value in fields.items():
            if value:
                conditions.append(f"json_extract(record, '$.{field}') = ?")
                values.append(value)
        for field, value in (("batch_id", batch_id), ("review_status", review_status)):
            if value:
                conditions.append(f"{field} = ?")
                values.append(value)
        if search:
            conditions.append("instr(id, ?) > 0")
            values.append(search)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connection() as db:
            summary = dict(db.execute(inventory + """SELECT COUNT(*) AS listings,
                COALESCE(SUM(review_status IN ('queued','running')),0) AS in_progress,
                COALESCE(SUM(review_status='completed'),0) AS completed FROM inventory""").fetchone())
            total = db.execute(inventory + "SELECT COUNT(*) FROM inventory" + where, values).fetchone()[0]
            rows = db.execute(inventory + "SELECT * FROM inventory" + where +
                              " ORDER BY batch_id DESC,id LIMIT ? OFFSET ?", [*values, limit, offset]).fetchall()
            options = {}
            for field in fields:
                options[field] = [row[0] for row in db.execute(
                    f"SELECT DISTINCT json_extract(record, '$.{field}') FROM listings ORDER BY 1")]
            options["batch_id"] = [row[0] for row in db.execute("SELECT DISTINCT batch_id FROM listings ORDER BY 1 DESC")]
        items = [{"batch_id": row["batch_id"], "listing": json.loads(row["record"]),
                  "prediction": json.loads(row["prediction"]), "review_status": row["review_status"]} for row in rows]
        return {"summary": summary, "options": options, "total": total, "items": items}

    def job_for_request(self, request_id):
        """Find an existing request before consulting current review configuration."""
        rows = self.query("SELECT id,listing_id FROM jobs WHERE request_id=?", (request_id,))
        return rows[0] if rows else None

    def create_job(self, listing_id, request_id, instructions, ai_settings, context_catalog=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT id,listing_id FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if existing:
                if existing["listing_id"] != listing_id:
                    raise Conflict("Request ID already belongs to another listing.")
                return existing["id"]
            row = db.execute("SELECT record,prediction FROM listings WHERE id=?", (listing_id,)).fetchone()
            if not row:
                raise KeyError(listing_id)
            snapshot = {"listing": json.loads(row["record"]), "prediction": json.loads(row["prediction"]),
                        "instructions": instructions, "ai_settings": ai_settings}
            if context_catalog is not None:
                snapshot["review_context"] = context_catalog.for_listing(snapshot["listing"])
            job_id = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,request_id,listing_id,status,snapshot) VALUES(?,?,?,'queued',?)",
                       (job_id, request_id, listing_id, json.dumps(snapshot)))
            return job_id

    def recover_reviews(self):
        # One API process only. In-memory background tasks do not survive restart.
        with self.connection() as db:
            return db.execute("UPDATE jobs SET status='failed',error='interrupted',updated=CURRENT_TIMESTAMP "
                              "WHERE status IN ('queued','running')").rowcount

    def claim_job(self, job_id):
        with self.connection() as db:
            return db.execute("UPDATE jobs SET status='running',updated=CURRENT_TIMESTAMP "
                              "WHERE id=? AND status='queued'", (job_id,)).rowcount == 1

    def job(self, job_id):
        rows = self.query("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not rows:
            raise KeyError(job_id)
        row = rows[0]
        for key in ["snapshot", "stages", "result"]:
            row[key] = json.loads(row[key]) if row[key] else None
        return row

    def save_stage(self, job_id, stage, result):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            stages = json.loads(db.execute("SELECT stages FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            stages[stage] = result
            db.execute("UPDATE jobs SET stages=?,updated=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(stages), job_id))

    def finish_job(self, job_id, result=None, error=None):
        with self.connection() as db:
            db.execute("UPDATE jobs SET status=?,result=?,error=?,updated=CURRENT_TIMESTAMP WHERE id=?",
                       ("failed" if error else "completed", json.dumps(result) if result is not None else None, error, job_id))
