import argparse
from engine import *
from models import *
from voc import *
from coco import *
from config import *
# seed_everything(config.seed)
import wandb
from backbones.config import config
from torchvision.transforms import RandAugment
from PIL import ImageDraw
from torch.optim import lr_scheduler

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

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
parser.add_argument('-b', '--batch-size', default=50, type=int,
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
parser.add_argument('--lr_scheduler',action='store_true', 
                    help='lr_schedule'),
parser.add_argument('--intermediate',action='store_true', 
                    help='intermediate'),

parser.add_argument('--dataset', default='voc', type=str)
parser.add_argument('--model', default='', type=str,
                    help='model name [resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, vit]')
parser.add_argument('--finetune', default="base", type=str,
                    help="finetuning model type [fc, gcn, sage, sa, transformer_encoder]")
parser.add_argument('--where', default=0, type=int)         
parser.add_argument('--optim_config', default=0, type=int)
parser.add_argument('--label_count', default=0, type=int,
                    help='only get data that contains this number of labels')

def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)


    use_gpu = torch.cuda.is_available()

    # define dataset
    if args.dataset=='voc':
        train_dataset = Voc2007Classification(args.data, 'trainval')
        val_dataset = Voc2007Classification(args.data, 'test')
        num_classes = 20
    elif args.dataset=='coco':
        train_dataset = COCO2014(args.data, phase='train', label_count=args.label_count)
        val_dataset = COCO2014(args.data, phase='val', )
        num_classes = 80

    resume = True if len(args.resume) else False
    if len(args.wandb) :
        wandb.init(project="ML-{}-{}-{}-{}".format(args.name, args.dataset, args.label_count, args.model), name="{}-{}-{}".format(args.wandb, args.finetune, args.seed), entity='seonghaeom', resume=resume)



    #exp1 model variants
    m_path = config[args.model]
    print("model path: {}".format(m_path))
    if args.model == 'resnet10':
        model = base_resnet10(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet18':
        model = base_resnet18(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet34':
        model = base_resnet34(num_classes=num_classes, pretrained=True)
    elif args.model == 'resnet50':
        model = base_resnet50(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    elif args.model == 'resnet101':
        model = base_resnet101(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    elif args.model == 'resnet152':
        model = base_resnet152(num_classes=num_classes, pretrained=True)
    elif args.model == 'vit':
        model = base_vit(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    elif args.model == 'swin':
        model = base_swin(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    elif args.model == 'swin_large':
        model = base_swin(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    elif args.model == 'convnext':
        model = base_convnext(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    elif args.model == 'mlpmixer':
        model = base_mlpmixer(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True, cond=args.intermediate, where=args.where)
    # elif args.model == 'vit-hybrid'
    #     model = base_vit_hybrid(model_path = m_path, num_classes=num_classes, image_size=args.image_size, pretrained=True)

    # exp2 load model
    adj_file = 'data/{}/{}_adj.pkl'.format(args.dataset, args.dataset)
    model = finetune_clf(model, args.finetune, num_classes=num_classes, adj_file=adj_file)

    # define loss function (criterion)
    if args.loss == "softmargin":
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss =="mse":
        criterion = nn.MSELoss()
    elif args.loss == 'asymmetric':
        from timm.loss import AsymmetricLossMultiLabel
        criterion = AsymmetricLossMultiLabel(gamma_pos=0,gamma_neg=0,eps=0.0)
    # define optimizer
    print(len(model.get_config_optim(args.lr, args.lrp)[args.optim_config:]))
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp)[args.optim_config:],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(params=model.get_config_optim(args.lr, args.lrp)[args.optim_config:], lr=args.lr, weight_decay=0)
    if args.lr_scheduler:
      scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataset), epochs=args.epochs, pct_start=0.2)
    else: scheduler = None

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
        state['wandb'] = args.wandb
    else:
        state['wandb'] = None
    
    state['name'] = args.name
    state['finetune'] = args.finetune
    state['model'] = args.model
    state['dataset'] = args.dataset

    normalize = transforms.Normalize(mean=model.image_normalization_mean, std=model.image_normalization_std)
    # state['train_transform'] = transforms.Compose([
    #                                   transforms.Resize((args.image_size, args.image_size)),
    #                                   CutoutPIL(cutout_factor=0.5),
    #                                   RandAugment(),
    #                                   transforms.ToTensor(),
    #                                   normalize,
    #                               ])
    engine = GCNMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_dataset, val_dataset, optimizer, scheduler)
    if len(args.wandb):
        wandb.log({"best_map": best_score["mAP"], "best_cf1": best_score["CF1"], "best_of1": best_score["OF1"] })



if __name__ == '__main__':
    main()
