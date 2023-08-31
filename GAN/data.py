"""
Prepare dataset for GAN training
Copyright (c) 2023 Global Health Labs, Inc
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import torchvision.datasets as dset
import torchvision.transforms as transforms
import cv2

__all__=['make_dataset','openCVDataset']

def make_dataset(dataset, dataroot, output_shape):
    """
    :param dataset: name of the data
    :return: pytorch dataset for DataLoader to utilize
    """
    dataset = openCVDataset(root= dataroot,output_shape=output_shape)
    # personalized dataset could be defined here
        
    assert dataset
    return dataset

class openCVDataset(dset.ImageFolder):
    def __init__(self, root, output_shape, is_valid_file=None):
        super(openCVDataset, self).__init__(root=root, is_valid_file=is_valid_file)
        self.output_shape = output_shape

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        # do your magic here
        if self.output_shape[2]==3:
            img = cv2.imread(image_path,cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED) # under the default setting that the img does not contain (black & white)
        # this has to be done CAREFULLY!
        img = cv2.resize(img, (self.output_shape[1], self.output_shape[0]))
        sample =  transforms.ToTensor()(img)
        if self.output_shape[2]==3:
            sample = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(sample)
        else:
            sample = transforms.Normalize((0.5), (0.5))(sample)
        
        return sample, target