import sys
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, scale, filename):
        super(Model, self).__init__()
        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 3
        self.scale = scale

        num_outputs = scale * scale * num_inputs
        status = self.file_reader(filename)
        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                self.IN,
                kernel_size,
                padding=kernel_size // 2))
        body.append(conv)

        for (IN, M1, M2) in status:
            body.append(Block(
                IN=IN,
                M1=M1,
                M2=M2,
                kernel_size=kernel_size,
                weight_norm=weight_norm)
            )

        conv = weight_norm(
            nn.Conv2d(
                self.IN,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))

        body.append(conv)
        self.body = nn.Sequential(*body)
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_outputs,
                skip_kernel_size,
                padding=skip_kernel_size // 2))

        self.skip = conv

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        x = x - self.image_mean
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        # batch_size, channels, in_height, in_width = x.size()
        
        
        
        # channels = channels / (self.scale * self.scale)
        # out = torch.split(x, self.scale * self.scale, dim=1)
        # out = [torch.reshape(a,(batch_size,1,self.scale,self.scale,in_height,in_width)) for a in out]
        # out = [torch.transpose(a,4,2) for a in out]
        # out = [torch.transpose(a,4,3) for a in out]
        # out = [torch.transpose(a,4,5) for a in out]
        # out = [torch.reshape(a,(batch_size,1,self.scale*in_height,self.scale*in_width)) for a in out]
        # out = torch.cat(out,1)

        # out = out + self.image_mean

        return x

    def file_reader(self, filename):

        with open(filename, 'r') as f:

            status = eval(f.readlines()[-1].replace('\n', ''))[1]
        self.IN = status[0][0]
        print(status)
        return status


class Block(nn.Module):
    def __init__(self, IN, M1, M2, kernel_size, weight_norm=torch.nn.utils.weight_norm):
        super(Block, self).__init__()
        body = []

        conv = weight_norm(nn.Conv2d(IN, M1, 1, padding=1 // 2))

        body.append(conv)
        body.append(nn.ReLU(inplace=True))

        conv = weight_norm(nn.Conv2d(M1, M2, 1, padding=1 // 2))

        body.append(conv)

        # conv = weight_norm(nn.Conv2d(M2, M2, 3, padding=3 // 2, groups=M2))
        # body.append(conv)

        conv = weight_norm(nn.Conv2d(M2, IN, kernel_size, padding=kernel_size // 2))
        body.append(conv)
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x


if __name__ == '__main__':
    if (len(sys.argv) != 4) and (len(sys.argv) != 5):
        print('Usage: python export_onnx.py <scale> <block_index.txt file path> <output_name> (<model.pt>)')
        sys.exit(1)
    scale = eval(sys.argv[1])
    filename = sys.argv[2]
    output_name = sys.argv[3]
    
    print(f'Create x{scale} SR model from {filename}')
    model = Model(scale=scale, filename=filename)
    
    # print(type(net))
    # print(len(net))
    # for k in net.keys():
    #     print(k)
    # exit(0)
    if len(sys.argv) == 5:
        net = torch.load(sys.argv[4])
        model.load_state_dict(net)
    
    
    dummy_input = torch.rand([1, 3, 360, 540])

    torch.onnx.export(model, dummy_input, f'{output_name}.onnx', input_names=['LR'], output_names=['HR'],
                      opset_version=9)
