# %%
from torch.utils.data import Dataset 
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):

        self.imagePaths = imagePaths#list of input image paths
        self.maskPaths = maskPaths  #path of ground truth
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB) #change color sequence BGR -> RGB
        mask = cv2.imread(self.maskPaths[idx], 0) #flag 0 is 'grayscale'

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)