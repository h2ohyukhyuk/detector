
import time
from .util import *
import argparse
from .darknet import Darknet
import pickle as pkl

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--path_img", dest="path_img", help="Video file to run detection on",
                        default="yolov3-from-scratch/images/dog-cycle-car.png", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="yolov3-from-scratch/cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3-from-scratch/data/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("yolov3-from-scratch/data/coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

classes = load_classes('yolov3-from-scratch/data/coco.names')
colors = pkl.load(open("yolov3-from-scratch/data/pallete", "rb"))
#print(len(colors))

# Detection phase
frame = cv2.imread(args.path_img)
img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
im_dim = frame.shape[1], frame.shape[0]
im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

if CUDA:
    im_dim = im_dim.cuda()
    img = img.cuda()

with torch.no_grad():
    output = model(img, CUDA)

start = time.time()
with torch.no_grad():
    output = model(img, CUDA)

    #print('model out shape: ', output.shape)
output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)
#print('write result shape: ', output.shape)

im_dim = im_dim.repeat(output.size(0), 1)
scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

# list(map(lambda x: write(x, frame), output))
results = output.cpu().numpy()
for result in results:
    draw_results(result, frame, classes, colors)

cv2.imshow("frame", frame)
key = cv2.waitKey()

print(time.time() - start)
print("inference time {:5.2f}".format(time.time() - start))
