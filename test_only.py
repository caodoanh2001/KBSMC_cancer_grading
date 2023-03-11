import os
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
from imgaug import augmenters as iaa
from misc.train_ultils_all_iter import *
import importlib


from loss.mtmr_loss import get_loss_mtmr
from loss.rank_ordinal_loss import cost_fn
from loss.dorn_loss import OrdinalLoss
import dataset as dataset
from config import Config
from loss.ceo_loss import CEOLoss, FocalLoss, SoftLabelOrdinalLoss, FocalOrdinalLoss, count_pred
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

####

class Tester(Config):
    def __init__(self, _args=None):
        super(Tester, self).__init__(_args=_args)
        if _args is not None:
            self.__dict__.update(_args.__dict__)
            print(self.run_info)
    
    ####
    def infer_step(self, net, batch, device):
        net.eval()  # infer mode

        imgs, true = batch
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long()  # not one-hot

        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            out_net = net(imgs)  # a list contains all the out put of the network
            if "CLASS" in self.task_type:
                logit_class = out_net
                prob = nn.functional.softmax(logit_class, dim=-1)
                return dict(logit_c=prob.cpu().numpy(),  # from now prob of class task is called by logit_c
                            true=true.cpu().numpy())

            if "REGRESS" in self.task_type:
                if "rank_ordinal" in self.loss_type:
                    logits, probas = out_net[0], out_net[1]
                    predict_levels = probas > 0.5
                    pred = torch.sum(predict_levels, dim=1)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                if "rank_dorn" in self.loss_type:
                    pred, softmax = net(imgs)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                if "soft_label" in self.loss_type:
                    logit_regress = (self.nr_classes - 1) * out_net
                    return dict(logit_r=logit_regress.cpu().numpy(),
                                true=true.cpu().numpy())
                if "FocalOrdinal" in self.loss_type:
                    logit_regress = out_net
                    pred = count_pred(logit_regress)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                else:
                    logit_regress = out_net
                    return dict(logit_r=logit_regress.cpu().numpy(),
                                true=true.cpu().numpy())

            if "MULTI" in self.task_type:
                logit_class, logit_regress = out_net[0], out_net[1]
                prob = nn.functional.softmax(logit_class, dim=-1)
                return dict(logit_c=prob.cpu().numpy(),
                            logit_r=logit_regress.cpu().numpy(),
                            true=true.cpu().numpy())
            

    ####
    def run_once(self, data_root_dir, fold_idx=None):
        log_dir = self.log_dir
        check_manual_seed(self.seed)
        #_, _, test_pairs = getattr(dataset, ('prepare_%s_data_test_2' % self.dataset))(data_root_dir)
        _, _, test_pairs = getattr(dataset, ('prepare_%s_data_test_1' % self.dataset))(data_root_dir)
        
        # --------------------------- Dataloader

        infer_augmentors = self.infer_augmentors()  # HACK at has_aux
        test_dataset = dataset.DatasetSerial(test_pairs[:], has_aux=False,
                                             shape_augs=iaa.Sequential(infer_augmentors[0]))


        test_loader = data.DataLoader(test_dataset,
                                      num_workers=self.nr_procs_valid,
                                      batch_size=self.infer_batch_size,
                                      shuffle=False, drop_last=False)

        device = 'cuda'

        # Define your network here
        # # # Note: this code for EfficientNet B0
        net_def = importlib.import_module('model_lib.efficientnet_pytorch.model')  # dynamic import
        
        if "FocalOrdinal" in self.loss_type:
            net = net_def.jl_efficientnet(task_mode='class', pretrained=True, num_classes=3)

        elif "rank_ordinal" in self.loss_type:
            net_def = importlib.import_module('model_lib.efficientnet_pytorch.model_rank_ordinal')  # dynamic import
            net = net_def.jl_efficientnet(task_mode='regress_rank_ordinal', pretrained=True)

        elif "mtmr" in self.loss_type:
            net_def = importlib.import_module('model_lib.efficientnet_pytorch.model_mtmr')  # dynamic import
            net = net_def.jl_efficientnet(task_mode='multi_mtmr', pretrained=True)

        elif "rank_dorn" in self.loss_type:
            net_def = importlib.import_module('model_lib.efficientnet_pytorch.model_rank_ordinal')  # dynamic import
            net = net_def.jl_efficientnet(task_mode='regress_rank_dorn', pretrained=True)

        else:
            net = net_def.jl_efficientnet(task_mode=self.task_type.lower(), pretrained=True)

        PATH_model = './log_colon_tma_20230311_att_ce_huber_rl/MULTI_ce_mse_cancer_Effi_seed5_BS64/_net_100.pth'
        net = torch.nn.DataParallel(net).to(device)
        checkpoint = torch.load(PATH_model)
        net.load_state_dict(checkpoint)

        net.eval()
        logits_c = []
        logits_r = []
        trues = []

        # Evaluating
        with tqdm(desc='Epoch %d - evaluation', unit='it', total=len(test_loader)) as pbar:
            for it, (images, gts) in enumerate(iter(test_loader)):
                results = self.infer_step(net, (images, gts), device)
                logits_c.append(results['logit_c'])
                logits_r.append(results['logit_r'])
                trues.append(results['true'])
                pbar.update()

        logits_c = np.concatenate(logits_c, axis=0)
        logits_r = np.concatenate(logits_r, axis=0)
        trues = np.concatenate(trues)
        preds_c = np.argmax(logits_c, axis=-1)

        if max(trues) == 4:
            trues -= 1

        print('----------------------------- Predictions by classification head -----------------------------')
        # import pdb; pdb.set_trace()
        print('Precision: ', precision_score(preds_c, trues, average='macro'))
        print('Recall: ', recall_score(preds_c, trues, average='macro'))
        print('F1: ', f1_score(preds_c, trues, average='macro'))
        print('Accuracy: ', accuracy_score(preds_c, trues))
        print('Confusion matrix: ')
        print(confusion_matrix(preds_c, trues))

        return

    ####
    def run(self, data_root_dir=None):
        self.run_once(data_root_dir, self.fold_idx)
        return

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run_info', type=str, default='REGRESS_rank_dorn',
                        help='CLASS, REGRESS, MULTI + loss, '
                             'loss ex: MULTI_mtmr, REGRESS_rank_ordinal, REGRESS_rank_dorn'
                             'REGRESS_FocalOrdinalLoss, REGRESS_soft_ordinal')
    parser.add_argument('--dataset', type=str, default='colon_tma', help='colon_tma, prostate_uhu')
    # parser.add_argument('--data_root_dir', type=str, default='../../datasets/KBSMC_colon_45wsis_cancer_grading_512_test_2/')
    parser.add_argument('--data_root_dir', type=str, default='../KBSMC_colon_tma_cancer_grading_512/')
    parser.add_argument('--seed', type=int, default=5, help='number')
    parser.add_argument('--alpha', type=int, default=5, help='number')
    parser.add_argument('--log_path', type=str, default='./log_colon_tma_20230307/')

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    tester = Tester(_args=args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    tester.run(data_root_dir=args.data_root_dir)