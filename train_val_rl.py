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
import torchvision


from loss.mtmr_loss import get_loss_mtmr
from loss.rank_ordinal_loss import cost_fn
from loss.dorn_loss import OrdinalLoss
import dataset as dataset
from config import Config
from loss.ceo_loss import CEOLoss, FocalLoss, SoftLabelOrdinalLoss, FocalOrdinalLoss, count_pred, inverse_huber_loss
from sklearn.metrics import recall_score
####

class Trainer(Config):
    def __init__(self, _args=None):
        super(Trainer, self).__init__(_args=_args)
        if _args is not None:
            self.__dict__.update(_args.__dict__)
            print(self.run_info)

    ####
    def view_dataset(self, mode='train', data_root_dir=None):
        train_pairs, valid_pairs = getattr(dataset, ('prepare_%s_data' % self.dataset))(data_root_dir=data_root_dir)
        if mode == 'train':
            train_augmentors = self.train_augmentors()
            ds = dataset.DatasetSerial(train_pairs, has_aux=False,
                                       shape_augs=iaa.Sequential(train_augmentors[0]),
                                       input_augs=iaa.Sequential(train_augmentors[1]))
        else:
            infer_augmentors = self.infer_augmentors()  # HACK
            ds = dataset.DatasetSerial(valid_pairs, has_aux=False,
                                       shape_augs=iaa.Sequential(infer_augmentors)[0])
        dataset.visualize(ds, 4)
        return

    ####
    def train_step(self, engine, net, net_baseline, batch, iters, scheduler, optimizer, device):
        net.train()  # train mode

        imgs_cpu, true_cpu = batch
        imgs_cpu = imgs_cpu.permute(0, 3, 1, 2)  # to NCHW
        scheduler.step(engine.state.epoch + engine.state.iteration / iters)  # scheduler.step(epoch + i / iters)
        # push data to GPUs
        imgs = imgs_cpu.to(device).float()
        true = true_cpu.to(device).long()  # not one-hot

        # -----------------------------------------------------------
        net.zero_grad()  # not rnn so not accumulate
        out_net = net(imgs)  # a list contains all the output of the network (regression + classification)
        loss = 0.

        logit_class , _ = out_net
        prob = F.softmax(logit_class, dim=-1)
        pred = torch.argmax(prob, dim=-1).detach().numpy()

        out_net_baseline = net_baseline(imgs)
        logit_class_baseline , _ = out_net_baseline
        prob_baseline = F.softmax(logit_class_baseline, dim=-1)
        pred_baseline = torch.argmax(prob_baseline, dim=-1).detach().numpy()

        import pdb; pdb.set_trace()

        acc = torch.mean((pred == true).float())  # batch accuracy
        # gradient update
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------
        return dict(
            loss=loss.item(),
            acc=acc.item(),
        )

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
    def run_once(self, data_root_dir, fold_idx):
        log_dir = self.log_dir
        check_manual_seed(self.seed)
        train_pairs, valid_pairs, test_pairs = getattr(dataset, ('prepare_%s_data' % self.dataset))(data_root_dir)
        # --------------------------- Dataloader

        train_augmentors = self.train_augmentors()
        train_dataset = dataset.DatasetSerial(train_pairs[:], has_aux=False,
                                              shape_augs=iaa.Sequential(train_augmentors[0]),
                                              input_augs=iaa.Sequential(train_augmentors[1]))

        infer_augmentors = self.infer_augmentors()  # HACK at has_aux
        infer_dataset = dataset.DatasetSerial(valid_pairs[:], has_aux=False,
                                              shape_augs=iaa.Sequential(infer_augmentors[0]))
        test_dataset = dataset.DatasetSerial(test_pairs[:], has_aux=False,
                                             shape_augs=iaa.Sequential(infer_augmentors[0]))

        train_loader = data.DataLoader(train_dataset,
                                       num_workers=self.nr_procs_train,
                                       batch_size=self.train_batch_size,
                                       shuffle=True, drop_last=True)
        valid_loader = data.DataLoader(infer_dataset,
                                       num_workers=self.nr_procs_valid,
                                       batch_size=self.infer_batch_size,
                                       shuffle=False, drop_last=False)
        test_loader = data.DataLoader(test_dataset,
                                      num_workers=self.nr_procs_valid,
                                      batch_size=self.infer_batch_size,
                                      shuffle=False, drop_last=False)

        # --------------------------- Training Sequence

        if self.logging:
            check_log_dir(log_dir)

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

        net = torch.nn.DataParallel(net).to(device)

        PATH_model = './_net_1950.pth'
        net_baseline = net
        checkpoint = torch.load(PATH_model)
        net_baseline.load_state_dict(checkpoint)

        for param in net_baseline.parameters():
            param.requires_grad = False

        net_baseline.eval()

        # optimizers
        optimizer = optim.AdamW(net.parameters(), lr=self.init_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.nr_epochs // 3, T_mult=1,
                                                                   eta_min=self.init_lr * 0.1, last_epoch=-1)
        
        iters = self.nr_epochs * self.epoch_length
        trainer = Engine(lambda engine, batch: self.train_step(engine, net, net_baseline, batch, iters, scheduler, optimizer, device))
        valider = Engine(lambda engine, batch: self.infer_step(net, batch, device))
        test = Engine(lambda engine, batch: self.infer_step(net, batch, device))

        # assign output
        if "CLASS" in self.task_type:
            infer_output = ['logit_c', 'true']
        if "REGRESS" in self.task_type:
            infer_output = ['logit_r', 'true']
        if "MULTI" in self.task_type:
            infer_output = ['logit_c', 'logit_r', 'pred_c', 'pred_r', 'true']

        ##
        events = Events.EPOCH_COMPLETED
        if self.logging:
            @trainer.on(events)
            def save_chkpoints(engine):
                torch.save(net.state_dict(), self.log_dir + '/_net_' + str(engine.state.iteration) + '.pth')

        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(valider, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(test, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        # TODO: refactor this
        RunningAverage(alpha=0.95, output_transform=lambda x: x['acc']).attach(trainer, 'acc')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['loss']).attach(trainer, 'loss')

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=['loss'])
        pbar.attach(valider)
        pbar.attach(test)

        # writer for tensorboard logging
        tfwriter = None  # HACK temporary
        if self.logging:
            tfwriter = SummaryWriter(logdir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file)  # create empty file

        ### TODO refactor again
        log_info_dict = {
            'logging': self.logging,
            'optimizer': optimizer,
            'tfwriter': tfwriter,
            'json_file': json_log_file if self.logging else None,
            'nr_classes': self.nr_classes,
            'metric_names': infer_output,
            'infer_batch_size': self.infer_batch_size  # too cumbersome
        }
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  lambda engine: scheduler.step(engine.state.epoch - 1))  # to change the lr
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_ema_results, log_info_dict)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, inference, valider, 'valid', valid_loader, log_info_dict)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, inference, test, 'test', test_loader, log_info_dict)
        valider.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        test.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)

        # Setup is done. Now let's run the training
        trainer.run(train_loader, self.nr_epochs, self.epoch_length)
        return

    ####
    def run(self, data_root_dir=None):
        if self.cross_valid:
            for fold_idx in range(0, trainer.nr_fold):
                trainer.run_once(fold_idx)
        else:
            # self.run_once(self.fold_idx)
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
    parser.add_argument('--data_root_dir', type=str, default='../KBSMC_colon_tma_cancer_grading_512/')
    parser.add_argument('--seed', type=int, default=5, help='number')
    parser.add_argument('--alpha', type=int, default=5, help='number')
    parser.add_argument('--log_path', type=str, default='./log_colon_tma_20230311_ce_huber/')

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    trainer = Trainer(_args=args)
    if args.view:
        trainer.view_dataset(data_root_dir=args.data_root_dir)
        exit()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    trainer.run(data_root_dir=args.data_root_dir)