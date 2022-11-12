

import argparse
def get_args(handle_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_bert', action='store_true', default=True)
    parser.add_argument('--with_no_text', action='store_true', default=False)
    parser.add_argument('--with_no_image', action='store_true', default=False)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--img_model', type=str)

    # Training setting
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--expid', type=str, default='test')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--save_model_dir', type=str)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--input_img_sz", type=int, default=1664,help = 'resnet-2048,densenet121-1024,densenet169-1664')
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])

    # Optimizer and scheduler setting
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta0', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)

    if handle_args:
        args = parser.parse_args(handle_args)
    else:
        args = parser.parse_args()

    return args