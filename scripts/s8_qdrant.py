"""Share the prepared S8 vectors without repeating corpus embedding."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from qdrant_client import QdrantClient, models

from ai_lib.retrieval.indexing import RetrievalIndex
from ai_lib.retrieval.qdrant import QdrantIndex
from lab_helpers.s8_assistant import index_directory, load_documents, raw_directory


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')


def export_index(embeddings, database, output, *, provenance, license_text):
    """Write plain records and float32 arrays, never Qdrant's internal database."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}. Choose another filename.")
    if not Path(database).is_dir():
        raise ValueError("No prepared Qdrant index found. Build or import it first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(database))
    created = False
    try:
        index = QdrantIndex(client, embeddings)
        state = index.open()
        manifest = {"format": "s8-qdrant-v1", "preparation": state,
                    "files": {}, "batches": []}
        with output.open('xb') as stream:
            created = True
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                def write(name, data):
                    archive.writestr(name, data)
                    manifest['files'][name] = hashlib.sha256(data).hexdigest()

                write('provenance.json', json_bytes(provenance))
                write('LICENSE.txt', license_text.encode('utf-8'))
                offset = None
                count = 0
                while True:
                    points, offset = client.scroll(index.collection_name, limit=512,
                        offset=offset, with_payload=True, with_vectors=True)
                    number = len(manifest['batches'])
                    prefix = f'batches/{number:06d}'
                    records = []
                    for point in points:
                        # Only document metadata travels with the vectors.
                        payload = {key: point.payload[key] for key in
                                   ['passage_id', 'window', 'start', 'end', 'url', 'title']}
                        records.append({"id": str(point.id), "payload": payload})
                    vectors = np.asarray([p.vector for p in points], dtype='<f4')
                    if not np.isfinite(vectors).all():
                        raise ValueError('The source collection contains invalid vectors.')
                    buffer = io.BytesIO()
                    np.save(buffer, vectors, allow_pickle=False)
                    write(prefix + '.npy', buffer.getvalue())
                    write(prefix + '.json', json_bytes(records))
                    manifest['batches'].append({"prefix": prefix, "count": len(points)})
                    count += len(points)
                    if number % 32 == 0:
                        print(f'Exported {count:,} vectors', flush=True)
                    if offset is None:
                        break
                if count != state['windows_completed']:
                    raise ValueError('The collection changed during export. Close other notebooks and retry.')
                archive.writestr('manifest.json', json_bytes(manifest))
        return count
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise
    finally:
        client.close()


def import_index(embeddings, source, destination):
    """Restore into a new folder; leave existing collections untouched."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('A local Qdrant folder already exists. It was left unchanged; move it aside before importing.')
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Invalid index archive: duplicate entries or missing manifest.')
        manifest = json.loads(archive.read('manifest.json'))
        if (not isinstance(manifest, dict) or not isinstance(manifest.get('batches'), list)
                or not manifest['batches'] or not isinstance(manifest.get('files'), dict)
                or not isinstance(manifest.get('preparation'), dict)):
            raise ValueError('Invalid index archive manifest.')
        expected = QdrantIndex(None, embeddings).configuration
        state = manifest.get('preparation', {})
        if (manifest.get('format') != 's8-qdrant-v1' or not state.get('complete')
                or state.get('configuration') != expected):
            raise ValueError('This ZIP does not match the prepared corpus and embedding settings.')
        allowed = {'manifest.json', 'LICENSE.txt', 'provenance.json'}
        for number, batch in enumerate(manifest['batches']):
            prefix = f'batches/{number:06d}'
            if (not isinstance(batch, dict) or batch.get('prefix') != prefix
                    or type(batch.get('count')) is not int or not 0 < batch['count'] <= 512):
                raise ValueError('Invalid batch entry in index archive.')
            allowed.update([prefix + '.npy', prefix + '.json'])
        if set(names) != allowed or set(manifest['files']) != allowed - {'manifest.json'}:
            raise ValueError('Unexpected files in index archive.')
        # Read members directly: no archive paths are extracted onto the machine.
        for name, digest in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'Archive checksum failed for {name}. Download the ZIP again.')

        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.qdrant-import-', dir=destination.parent) as temporary:
            staging = Path(temporary).resolve() / 'qdrant'
            assert staging.is_relative_to(destination.parent)
            client = QdrantClient(path=str(staging))
            try:
                index = QdrantIndex(client, embeddings)
                client.create_collection(index.collection_name,
                    vectors_config=models.VectorParams(size=expected['dimensions'], distance=models.Distance.COSINE),
                    metadata={"preparation": {**state, "complete": False}})
                seen = set()
                for number, batch in enumerate(manifest['batches']):
                    prefix = batch['prefix']
                    vectors = np.load(io.BytesIO(archive.read(prefix + '.npy')), allow_pickle=False)
                    records = json.loads(archive.read(prefix + '.json'))
                    if (vectors.dtype != np.dtype('<f4')
                            or vectors.shape != (batch['count'], expected['dimensions'])
                            or not isinstance(records, list) or len(records) != batch['count'] or not np.isfinite(vectors).all()
                            or (np.linalg.norm(vectors, axis=1) == 0).any()):
                        raise ValueError('Invalid vectors or records in index archive.')
                    points = []
                    for record, vector in zip(records, vectors):
                        if (not isinstance(record, dict) or not isinstance(record.get('id'), str)
                                or not isinstance(record.get('payload'), dict)):
                            raise ValueError('Invalid point record in index archive.')
                        point_id = record['id']
                        if point_id in seen:
                            raise ValueError('Duplicate point IDs in index archive.')
                        seen.add(point_id)
                        points.append(models.PointStruct(id=point_id, vector=vector.tolist(), payload=record['payload']))
                    client.upsert(index.collection_name, points=points, wait=True)
                    if number % 32 == 0:
                        print(f'Imported {len(seen):,}/{state["windows_completed"]:,} vectors', flush=True)
                # Tokenize to check window coverage; no corpus embedding takes place.
                embeddings.prepare_model()
                index._verify_coverage(state)
                index._save_state(state)
                index.open()
            finally:
                client.close()
            if destination.exists():
                raise FileExistsError('A Qdrant folder appeared during import; it was left unchanged.')
            staging.rename(destination)
    return state['windows_completed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('export', help='Create a Moodle delivery ZIP.').add_argument('--output', type=Path, required=True)
    actions.add_parser('import', help='Install a downloaded index ZIP.').add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        embeddings = RetrievalIndex(load_documents(root))
        folder = index_directory(root)
        embeddings.build_lexical(folder)
        print('Close other notebooks using Qdrant before continuing.', flush=True)
        if args.action == 'export':
            raw = raw_directory(root)
            count = export_index(embeddings, folder/'qdrant', args.output,
                provenance=json.loads((raw/'download.json').read_text(encoding='utf-8')),
                license_text=(raw/'LICENSE').read_text(encoding='utf-8'))
            print(f'Saved {count:,} vectors to {args.output}')
        else:
            count = import_index(embeddings, args.archive, folder/'qdrant')
            print(f'Ready: {count:,} vectors. Select retrieval_method="qdrant" in the notebook.')
    except (ValueError, OSError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
