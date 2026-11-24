"""Show one MCB part from several camera views, next to the dataset's point cloud for it.

    python render_cad_obj.py                                   # random part, in a window
    python render_cad_obj.py --cls "Right spur gears" --views 6
    python render_cad_obj.py --index 123 --split test --save part.png [--show]
"""
import argparse

import cv2

from cad_data_loader.cad_obj_dataset import MCB_ROOT, CadObjDataset
from cad_data_loader.cad_views import create_points_view, create_views, load_triangles, tile

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--split", default="train", choices=["train", "test"])
p.add_argument("--index", type=int, help="sample index (default: random)")
p.add_argument("--cls", help="random sample from this class folder")
p.add_argument("--views", type=int, default=4, help="cameras around the part")
p.add_argument("--elev", type=float, default=25, help="camera elevation, degrees")
p.add_argument("--show", action="store_true", help="open a window (default without --save)")
p.add_argument("--save", metavar="PATH.png", help="save the image (no window unless --show)")
args = p.parse_args()

ds = CadObjDataset(MCB_ROOT / args.split)
index = args.index if args.index is not None else ds.pick_index(args.cls)
points, label = ds[index]
title = f"{ds.classes[label]} ({ds.files[index].name}, #{index})"
print(title)

images = create_views(load_triangles(ds.files[index]), views=args.views, elev=args.elev)
images.append(create_points_view(points.numpy(), elev=args.elev))
grid = tile(images, cols=4)  # a grid, not one long strip: macOS pads very wide windows

if args.save:
    cv2.imwrite(args.save, grid)
if args.show or not args.save:
    scale = min(1.0, 1200 / grid.shape[1])      # fit on a laptop screen
    shown = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)  # window = image size, so no blank margins
    cv2.imshow(title, shown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
