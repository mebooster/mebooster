import torch


class DataRunner():
    def __init__(self):
        self.data_output_path = '..\\data'

    def run(self, args, device):
        #gaussain
        if args.data_type == 'GAUSSIAN':
            x_train = torch.randn([args.N_query, args.channel, args.width, args.width])  # chw
            torch.save(x_train, self.data_output_path+'\\x.pt')

            x_test = torch.randn([args.N_test, args.channel, args.width, args.width])  # chw
            torch.save(x_test, self.data_output_path+'\\x_test.pt')
            stride = 3
            kernel = 3
            step_1_x = x_train[:, :, :kernel, :kernel]
            step_2_x = x_train[:, :, :kernel, stride:]
            step_3_x = x_train[:, :, stride:, :kernel]
            step_4_x = x_train[:, :, stride:, stride:]

            torch.save(step_1_x, self.data_output_path+'\\step_1_x.pt')  # [batch_size, 1, 3, 3]
            torch.save(step_2_x, self.data_output_path+'\\step_2_x.pt')
            torch.save(step_3_x, self.data_output_path+'\\step_3_x.pt')
            torch.save(step_4_x, self.data_output_path+'\\step_4_x.pt')
        #elif args.data_type == 'MNIST':
            #save mnist data
        #elif args.data_type == 'GRSTB':
            #save grstb data
