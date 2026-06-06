import argparse
import config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline principal de Visión 3D")
    parser.add_argument('--quiet', action='store_true', help='Silenciar mensajes informativos')
    parser.add_argument('--no-gui', action='store_true', help='No abrir ventanas GUI')
    parser.add_argument('--outdir', type=str, default='output', help='Directorio donde guardar resultados')
    parser.add_argument('--refine-corners', action='store_true', help='Refinar esquinas ArUco')
    parser.add_argument('--ba-nfev', type=int, default=2000, help='Max eval para bundle adjustment')
    parser.add_argument('--min-corner-quality', type=float, default=0.0, help='Umbral relativo filtrado esquinas')
    parser.add_argument('--ransac-threshold', type=float, default=2.0, help='Umbral Sampson (px) para RANSAC')
    parser.add_argument('--filter-reproj', type=float, default=30.0, help='Umbral filtro reproyección')
    parser.add_argument('--usar-dino', action='store_true', help='Usar DINOv2 para matching semántico en lugar de SGBM')
    parser.add_argument('--plane-sweep', action='store_true', default=True, help='Usar Plane Sweeping Estéreo en lugar de SGBM (por defecto)')
    
    args, _ = parser.parse_known_args()

    if args.quiet:
        config.set_verbose(False)
    if args.no_gui:
        config.set_show_gui(False)
    
    outdir = args.outdir
    if not config.SHOW_GUI:
        outdir = config.ensure_output_dir(outdir)
        
    return args, outdir
