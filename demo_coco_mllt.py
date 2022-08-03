import argparse
from engine import *
from models import *
from coco import *
from util import *
from config import *
# seed_everything(config.seed)
import wandb


parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='', type=str,
                    help='model name [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, vit]')
parser.add_argument('--name', default='exp2', type=str,
                    help='wandb prj name')        
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--wandb', default='', type=str, 
                    help='logging with title at wandb')
parser.add_argument('--seed', default=42, type=int, 
                    help='seed everything'),

parser.add_argument('--gcn', action='store_true',
                    help='apply gcn learning method')
parser.add_argument('--inductive', action='store_true',
                    help='apply inductive learning method')
parser.add_argument('--transformer', action='store_true',
                    help='apply transformer method')
parser.add_argument('--base', action='store_true',
                    help='apply base fc method')
parser.add_argument('--transformer_encoder', action='store_true',
                    help='apply transformer encoder'),
                    
parser.add_argument('--t', default=0.4, type=float)
parser.add_argument('--p', default=0.25, type=float)
parser.add_argument('--optim_config', default=[], type=int, nargs='+',)
parser.add_argument('--LT', action='store_true',
                    help='longtail dataset')


def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    print(args)


    use_gpu = torch.cuda.is_available()

    train_dataset = COCO2014(args.data, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl', LT=args.LT)
    val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
    num_classes = 80

    if len(args.wandb) :
        wandb.init(project="ML-{}-COCO-{}-{}".format(args.name, args.LT, args.model), name="{}".format(args.wandb))

    #exp1 model variants
    if args.model == 'resnet10':
        model = base_resnet10(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet18':
        model = base_resnet18(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet34':
        model = base_resnet34(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet50':
        model = base_resnet50(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet101':
        model = base_resnet101(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet152':
        model = base_resnet152(num_classes=num_classes, pretrained=True)
    elif args.model == 'vit':
        model = base_vit(num_classes=num_classes, pretrained=True)

    # exp2 load model
    if args.inductive:
        assert args.model == ''
        model = sage_resnet101(num_classes=num_classes, t=args.t, p=args.p, pretrained=True, adj_file='data/coco/coco_adj.pkl')
    elif args.transformer:
        assert args.model == ''
        model = trans_resnet101(num_classes=num_classes, pretrained=True, adj_file='data/coco/coco_adj.pkl')
    elif args.base:
        assert args.model == ''
        model = base_resnet101(num_classes=num_classes, pretrained=True, adj_file='data/coco/coco_adj.pkl')
    elif args.transformer_encoder:
        assert args.model == ''
        model = trans_encoder_resnet101(num_classes=num_classes, pretrained=True, adj_file='data/coco/coco_adj.pkl')
    elif args.gcn:
        assert args.model == ''
        model = gcn_resnet101(num_classes=num_classes, t=args.t, pretrained=True, adj_file='data/coco/coco_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD([model.get_config_optim(args.lr, args.lrp)[i] for i in args.optim_config],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    scheduler = None

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/coco/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['inductive'] = args.inductive
    state['transformer'] = args.transformer
    if args.evaluate:
        state['evaluate'] = True
    if len(args.wandb):
        state['wandb'] = args.wandb
    else:
        state['wandb'] = None
    state['p'] = args.p
    state['tau'] = args.t
    engine = GCNMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_dataset, val_dataset, optimizer, scheduler)
    
    if len(args.wandb):
        wandb.log({"best_map": best_score["mAP"], "best_cf1": best_score["CF1"], "best_of1": best_score["OF1"] })

if __name__ == '__main__':
    main_coco()
