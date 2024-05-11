import os
import argparse
from datetime import datetime
import cv2
from flask import url_for
import torch
from torchvision.transforms import ToTensor, Resize
import numpy as np
from models import UNet  # Ensure your model import is correct
#from dataset import test_DIV2K_dataset  # Ensure correct import path
# from utils.conversion import YUV2RGB,RGB2YUV  # Check this utility function to ensure it's applicable


def RBB2Y(img):
    y = 16.  + (65.481 * img[:, :, 0]  + 128.553 * img[:, :, 1] +  24.966 * img[:, :, 2]) / 255.
    return y

def RGB2YUV(img):

    if len(img.shape) == 4:
        img = img.squeeze(0)

    y  = 16.  + (65.481 * img[:, :, 0]  + 128.553 * img[:, :, 1] +  24.966 * img[:, :, 2]) / 255.
    cb = 128. + (-37.797 * img[:, :, 0] -  74.203 * img[:, :, 1] + 112.000 * img[:, :, 2]) / 255.
    cr = 128. + (112.000 * img[:, :, 0] -  93.786 * img[:, :, 1] -  18.214 * img[:, :, 2]) / 255.

    if type(img) == np.ndarray:
        return np.array([y, cb, cr])
    elif type(img) == torch.Tensor:
        return torch.stack((y, cb, cr), 0)
    else:
        raise Exception("Conversion type not supported", type(img))


def YUV2RGB(img):
    r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
    g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
    b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
    return torch.stack([r, g, b], 0).permute(1, 2, 0)


class Orchestrator():
    def __init__(self, args):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.result_path = os.path.join(args.result_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # os.makedirs(self.result_path, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This will only create the 'output' directory if it doesn't exist.
        self.result_dir = args.result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        if args.model == "unet":
            self.model = UNet().to(self.device)
        else:
            raise ValueError("Model not available")

        self.model.load_state_dict(torch.load(args.model_path, map_location=self.device))
        self.model.eval()  # Set model to evaluation mode

    def test(self, image_path, filename):
        print(image_path)
        lr  = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) # Load and convert the low-resolution image to RGB
        #hr = cv2.cvtColor(cv2.imread(hr_image_path), cv2.COLOR_BGR2RGB) # Load and convert the high-resolution image to RGB

        lr = torch.from_numpy(lr) # Convert low-resolution image to PyTorch tensor
        #hr = torch.from_numpy(hr) # Convert high-resolution image to PyTorch tensor

        lr = RGB2YUV(lr) # Convert low-resolution image to YUV color space
        #hr = RGB2YUV(hr) # Convert high-resolution image to YUV color space

        lr = lr / 255.0 # Normalize low-resolution image
        #hr = hr / 255.0 # Normalize high-resolution image
        # Process the single image
        #lr, hr = lr.unsqueeze(0).to(self.device), hr.unsqueeze(0).to(self.device)  # Add batch dimension and send to device


        with torch.no_grad():
                input = lr.to(self.device)
                print("input-shape",input.shape)
                #print("target-shape",target.shape)

                input = input.unsqueeze(0)  # Adds a batch dimension at the start, changes shape from [C, H, W] to [N, C, H, W] where N=1
                #target = target.unsqueeze(0)
                input_channel = input[:, 0:1, :, :]
                #target_channel = target[:, 0:1, :, :]
                predict = self.model(input_channel)
                print(predict.shape)

                _, _, height, width = input.shape

                resizer = Resize([height, width])
                input_exp = resizer(input)

                print("Shapes before concatenation:")
                print("Predict shape:", predict.shape)
                print("Input_exp shape:", input_exp.shape)


                # Resize the predict tensor to match input_exp's size in dimension 1
                predict_resized = torch.nn.functional.interpolate(predict, size=input_exp.shape[2:], mode='nearest')

                print("Predict_resized shape:", predict_resized.shape)

                # Concatenate the resized predict tensor with input_exp
                predict = torch.cat((predict_resized, input_exp[:, 1:2, :, :], input_exp[:, 2:3, :, :]), 1)

                self.complete_test(input_exp, predict, filename)

    def complete_test(self, input, predict, filename):
        input = input.squeeze(0) * 255
        #target = target.squeeze(0) * 255
        predict = predict.squeeze(0) * 255

        input = YUV2RGB(input)
        predict = YUV2RGB(predict)
        #target = YUV2RGB(target)

        input = input.clip(0, 255).cpu().detach().numpy().astype(np.uint8)
        #target = target.clip(0, 255).cpu().detach().numpy().astype(np.uint8)
        predict = predict.clip(0, 255).cpu().detach().numpy().astype(np.uint8)

        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Create the output filename with timestamp
        output_filename = f"{timestamp}.jpg"
        output_path = os.path.join(self.result_dir, output_filename)

        # output_path = os.path.join(self.result_path, filename)
        cv2.imwrite(output_path, cv2.cvtColor(predict, cv2.COLOR_RGB2BGR))
        print(f"Output saved to {output_path}")

def process_image(image_path, model_path='models/best_train_model.pt'):
    args = argparse.Namespace()
    args.result_dir = 'static/output'
    args.model_path = model_path
    args.model = 'unet'

    orchestrator = Orchestrator(args)
    filename = create_jpg_filename()
    orchestrator.test(image_path, filename)

    return os.path.join(orchestrator.result_dir, filename)

def create_jpg_filename():
    # Get the current date and time
    now = datetime.now()
    # Format as a string, for example '2024-05-08-15-23-45'
    formatted_date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    # Create the file name with the .jpg extension
    file_name = f"{formatted_date_time}.jpg"
    # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # filename = f"{timestamp}/{timestamp}.jpg"
    # image_url = url_for('output', filename=filename)
    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='Model type')
    #parser.add_argument('--lr_image_path', default = "/home/016651206/project/test_single_inference/LR/0_lr.jpg", type=str, help='Path to the low-resolution image file')
    parser.add_argument('--image_path', default = "images/0.jpg", type=str, help="Path to the Test Image")
    #parser.add_argument('--hr_image_path', default = "/home/016651206/project/test_single_inference/HR/0_hr.jpg", type=str, help='Path to the high-resolution image file')
    #parser.add_argument('--test_data', nargs=2, default=('custom_dataset_generation/HR', 'custom_dataset_generation/LR'), help='path to test directory')
    parser.add_argument('--result_dir', type=str, default='test_single_results', help='results directory')
    parser.add_argument('--model_path', type=str, default='models/best_train_model.pt', help='test only')

    args = parser.parse_args()

    orch = Orchestrator(args)

    file_name = create_jpg_filename()
    #file_name = "0.jpg"

    start_time = datetime.now()
    #lr_image_path, hr_image_path = orch.create_hr_lr_image(args.image_path, args.lr_image_path, args.hr_image_path)
    orch.test(args.image_path, filename=file_name)
    end_time = datetime.now() - start_time
    print("End Time:- ", end_time.total_seconds())