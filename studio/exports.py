"""Prepare mesh downloads without changing the generated source files."""
import os
import tempfile
import zipfile
from pathlib import Path

from .storage import validate_glb


def export_model(model_dir, export_dir, file_format='glb', include_texture=True, max_faces=None):
    import trimesh

    model_dir, export_dir = Path(model_dir), Path(export_dir)
    file_format = file_format.lower()
    if file_format not in {'glb', 'obj', 'ply', 'stl'}:
        raise ValueError('Choose GLB, OBJ, PLY or STL.')
    if include_texture and file_format not in {'glb', 'obj'}:
        raise ValueError('Use GLB or OBJ to include texture maps.')
    if max_faces is not None and (not isinstance(max_faces, int) or max_faces < 4):
        raise ValueError('The target must be at least four faces.')

    source = model_dir/('textured_mesh.glb' if include_texture else 'white_mesh.glb')
    validate_glb(source, textured=include_texture)
    mesh = trimesh.load(source, force='mesh', process=False)
    export_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.export-', dir=export_dir) as temp:
        working = Path(temp)
        if max_faces is not None and len(mesh.faces) > max_faces:
            import pymeshlab

            original_material = mesh.visual.material if include_texture else None
            mesh.export(working/'source.obj')
            meshes = pymeshlab.MeshSet()
            meshes.load_new_mesh(str(working/'source.obj'))
            if include_texture:
                meshes.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=max_faces)
            else:
                meshes.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
            meshes.save_current_mesh(str(working/'simplified.obj'), save_textures=False)
            mesh = trimesh.load(working/'simplified.obj', force='mesh', process=False)
            if include_texture:
                if getattr(mesh.visual, 'uv', None) is None:
                    raise ValueError('Simplification did not retain UV coordinates.')
                mesh.visual.material = original_material

        if not include_texture:
            mesh.visual = trimesh.visual.ColorVisuals()
        kind = 'textured' if include_texture else 'shape'
        name = f'{model_dir.name}_{kind}_{len(mesh.faces)}'
        bundle = working/'bundle'
        bundle.mkdir()
        output = bundle/(name+'.'+file_format)
        mesh.export(output)
        if file_format == 'glb':
            validate_glb(output, textured=include_texture)
        elif not len(trimesh.load(output, force='mesh', process=False).faces):
            raise ValueError('The exported mesh has no triangles.')

        if file_format == 'obj' and include_texture:
            # OBJ uses companion material and image files, so download them together.
            # Wavefront PBR extensions retain the GLB's roughness and metallic maps.
            material = mesh.visual.material
            packed = getattr(material, 'metallicRoughnessTexture', None)
            if packed is not None:
                channels = packed.convert('RGB').split()
                channels[1].save(bundle/'roughness.png')
                channels[2].save(bundle/'metallic.png')
                roughness = getattr(material, 'roughnessFactor', None)
                metallic = getattr(material, 'metallicFactor', None)
                for mtl in bundle.glob('*.mtl'):
                    with mtl.open('a', encoding='utf-8') as stream:
                        stream.write(f'\nPr {roughness if roughness is not None else 1.0}\n'
                                     f'Pm {metallic if metallic is not None else 1.0}\n'
                                     'map_Pr roughness.png\nmap_Pm metallic.png\n')
            archive = working/(name+'.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
                for path in sorted(bundle.iterdir()):
                    if path.is_file():
                        zipped.write(path, path.name)
            output = archive
        target = export_dir/output.name
        os.replace(output, target)
    return target
