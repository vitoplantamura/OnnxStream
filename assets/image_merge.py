# requirements: Numpy, Pillow
# !!!
# correct images paths below
# !!!
# correct font path below

from PIL import Image, ImageDraw, ImageFont
import numpy as np

THUMBNAIL_SIZE = 256

SAMPLERS = []
# uncomment next line to render by halves
#'''
SAMPLERS += [
"euler",
"tcd",
"ddim",
"dpm++2mv2",
"dpm++2m",
"heun",
"dpm2",
"lms",
"ipndm_vo",
"dpm++3msde",
"ipndm_v",
]
#'''
# uncomment next line to render by halves
#'''
SAMPLERS += [
"ipndm",
"taylor3",
"ddpm",
"ddpm_a",
"dpm++2s",
"lcm",
"tcd_a",
"ddim_a",
"euler_a",
"dpm++2s_a",
"dpm++3msde_a",
]
#'''

NUMBER_OF_SAMPLERS = len(SAMPLERS)
print(f"Making grid {THUMBNAIL_SIZE}x{THUMBNAIL_SIZE} x {NUMBER_OF_SAMPLERS} x 3: ", end='')

gi = np.zeros((THUMBNAIL_SIZE * 3 + 5 * 2,
               THUMBNAIL_SIZE * NUMBER_OF_SAMPLERS + 5 * NUMBER_OF_SAMPLERS - 5,
               3), np.uint8)
gi[...] = 127   # grey background

def place_image(gi, y, n):
    i = Image.open(n).convert('RGB')
    s = i.size
    if s[0] != THUMBNAIL_SIZE or s[1] != THUMBNAIL_SIZE:
        i = i.resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
    gi[y : y + THUMBNAIL_SIZE, x : x + THUMBNAIL_SIZE, :] = np.array(i)

def draw_with_shadow(i, x, y, t, c, f):
    ImageDraw.Draw(i).text((x + 2, y + 2), t, (0,0,0), f)
    ImageDraw.Draw(i).text((x,     y    ), t, c,       f)


grid_name = "./i/grid.jpg"

x = 0
for sampler in SAMPLERS:
    y = 0
    place_image(gi, y, f"./i/sd/test_50_{sampler}_onnxstream_org.jpg")
    y += THUMBNAIL_SIZE + 5
    place_image(gi, y, f"./i/turbo/test_turbo_15_{sampler}_onnxstream_orgo.jpg")
    y += THUMBNAIL_SIZE + 5
    place_image(gi, y, f"./i/xl/test_xl_50_{sampler}_onnxstream_org.jpg")

    print(f"{sampler} ", end='')
    x += THUMBNAIL_SIZE + 5
print()

gi = Image.fromarray(gi)
font = ImageFont.truetype('/system/fonts/SourceSansPro-Bold.ttf', size = int(THUMBNAIL_SIZE / 8.5))
x = 0
for sampler in SAMPLERS:
    draw_with_shadow(gi, x + 10, 5, sampler, (255,255,255), font)
    x += THUMBNAIL_SIZE + 5

y = int(THUMBNAIL_SIZE * 0.87) - 5
draw_with_shadow(gi, 10, y, '50 steps, SD 1.5', (255,255,255), font)
y += THUMBNAIL_SIZE + 5
draw_with_shadow(gi, 10, y, '15 steps, SDXL Turbo', (255,255,255), font)
y += THUMBNAIL_SIZE + 5
draw_with_shadow(gi, 10, y, '50 steps, SDXL', (255,255,255), font)

try:
    print('Saving "' + grid_name + '".')
    gi.save(grid_name, quality=97)
except:
    print('FAILED saving "' + grid_name + '", read-only storage?')
