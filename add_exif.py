"""
渲染完成后运行此脚本，给 my_data_meshroom/ 里的图片写入相机 EXIF。
用系统 Python 运行：python add_exif.py
"""
import piexif, os, glob

IMG_DIR  = os.path.join(os.path.dirname(__file__), "my_data_meshroom")
IMG_W, IMG_H = 1920, 1080

files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
if not files:
    print(f"未找到图片：{IMG_DIR}")
    exit(1)

exif_dict = {
    "0th": {
        piexif.ImageIFD.Make:  b"Blender",
        piexif.ImageIFD.Model: b"VirtualCamera",
    },
    "Exif": {
        piexif.ExifIFD.FocalLength:           (27, 1),   # 27mm
        piexif.ExifIFD.FocalLengthIn35mmFilm:  27,
        piexif.ExifIFD.PixelXDimension:        IMG_W,
        piexif.ExifIFD.PixelYDimension:        IMG_H,
    }
}
exif_bytes = piexif.dump(exif_dict)

for path in files:
    piexif.insert(exif_bytes, path)

print(f"完成：{len(files)} 张图已写入 EXIF（焦距 27mm，{IMG_W}×{IMG_H}）")
