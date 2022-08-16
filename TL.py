import argparse
from engine import *
from models import *
from voc import *
from coco import *
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
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40,80], type=int, nargs='+',
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--name', default='exp', type=str,
                    help='wandb prj name')
parser.add_argument('--wandb', default='', type=str, 
                    help='logging with title at wandb')
parser.add_argument('--seed', default=42, type=int, 
                    help='seed everything'),
parser.add_argument('--loss', default='softmargin', type=str, 
                    help='loss'),

parser.add_argument('--dataset', default='voc', type=str)
parser.add_argument('--model', default='', type=str,
                    help='model name [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, vit]')
parser.add_argument('--finetune', default="base", type=str,
                    help="finetuning model type [fc, gcn, sage, sa, transformer_encoder]")
parser.add_argument('--num_block', default=4, type=int,
                    help="number of transformer block")
parser.add_argument('--num_head', default=4, type=int,
                    help="number of attention head")
                    
parser.add_argument('--t', default=0.4, type=float)
parser.add_argument('--p', default=0.25, type=float)
parser.add_argument('--optim_config', default=4, type=int)
parser.add_argument('--LT', action='store_true',
                    help='longtail dataset')

def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)



    use_gpu = torch.cuda.is_available()

    # define dataset
    if args.dataset=='voc':
        train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl', LT=args.LT)
        val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
        num_classes = 20
    elif args.dataset=='coco':
        train_dataset = COCO2014(args.data, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl', LT=args.LT)
        val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl')
        num_classes = 80

    if len(args.wandb) :
        if args.finetune=='transformer_encoder':
            wandb.init(project="ML-{}-{}-{}-{}".format(args.name, args.dataset, args.LT, args.model), name="{}-{}-{}-{}-{}".format(args.wandb, args.finetune, args.num_block, args.num_head, args.seed))
        else:
            wandb.init(project="ML-{}-{}-{}-{}".format(args.name, args.dataset, args.LT, args.model), name="{}-{}-{}".format(args.wandb, args.finetune, args.seed))

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
    if args.LT:
        adj_file = '{}_lt_adj.pkl'.format(args.dataset)
    else: adj_file = 'data/{}/{}_adj.pkl'.format(args.dataset, args.dataset)
    model = finetune_clf(model, args.finetune, num_classes=num_classes, num_block=args.num_block, num_head = args.num_head, adj_file=adj_file)

    # define loss function (criterion)
    if args.loss == "softmargin":
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss =="mse":
        criterion = nn.MSELoss()
    # define optimizer
    print(len(model.get_config_optim(args.lr, args.lrp)[args.optim_config:]))
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp)[args.optim_config:],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = None

    if args.model=='vit':
        args.image_size = 224
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
            'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/{}/'.format(args.dataset)
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    if len(args.wandb):
        state['wandb'] = "{}-{}-{}-{}".format(args.wandb, args.num_block, args.num_head, args.seed)
        
    else:
        state['wandb'] = None
    
    state['name'] = args.name
    state['p'] = args.p
    state['tau'] = args.t
    state['finetune'] = args.finetune
    state['model'] = args.model
    state['dataset'] = args.dataset
    state['LT'] = args.LT
    engine = GCNMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_dataset, val_dataset, optimizer, scheduler)
    if len(args.wandb):
        wandb.log({"best_map": best_score["mAP"], "best_cf1": best_score["CF1"], "best_of1": best_score["OF1"] })



if __name__ == '__main__':
    main()
