import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 96, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
#        self.conv4_1 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))

        self.conv5 = nn.Conv2d(1, 96, (5, 5), (1, 1), (2, 2))
        self.conv6 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
#        self.conv8_1 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))

        self.conv9 = nn.Conv2d(1, 96, (7, 7), (1, 1), (3, 3))
        self.conv10 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv11 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv12 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
#        self.conv12_1 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))

        self.conv13 = nn.Conv2d(1, 96, (9, 9), (1, 1), (3, 3))
        self.conv14 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv15 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))
        self.conv16 = nn.Conv2d(96, 96, (3, 3), (1, 1), (1, 1))


        self.conv_end = nn.Conv2d(384, 1, (3, 3), (1, 1), (1, 1))




        self.pixel_shuffle = nn.PixelShuffle(1)

        self._initialize_weights()

    def forward(self, x):
        src_x = x
        x1 = self.relu(self.conv1(src_x))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = self.relu(self.conv4(x1))
#        x1 = self.relu(self.conv4_1(x1))



        x2 = self.relu(self.conv5(src_x)) 
        x2 = self.relu(self.conv6(x2))
        x2 = self.relu(self.conv7(x2))
        x2 = self.relu(self.conv8(x2))
#        x2 = self.relu(self.conv8_1(x2))

        x3 = self.relu(self.conv9(src_x))
        x3 = self.relu(self.conv10(x3))
        x3 = self.relu(self.conv11(x3))
        x3 = self.relu(self.conv12(x3))
#        x3 = self.relu(self.conv12_1(x3))

#        x4 = self.relu(self.conv13(src_x))
        x4 = self.relu(self.conv14(x3))
        x4 = self.relu(self.conv15(x3))
        x4 = self.relu(self.conv16(x3))

        
        x = torch.cat([x1,x2,x3,x4], 1)
        x = self.pixel_shuffle(self.conv_end(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
#        init.orthogonal_(self.conv4_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv6.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv7.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv8.weight, init.calculate_gain('relu'))
#        init.orthogonal_(self.conv8_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv9.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv10.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv11.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv12.weight, init.calculate_gain('relu'))
#        init.orthogonal_(self.conv12_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv13.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv14.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv15.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv16.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv_end.weight, init.calculate_gain('relu'))

