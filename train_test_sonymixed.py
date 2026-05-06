import argparse
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np


from model.NetWork3 import DUNF 
from Losses.color_loss import ColorHistogramKLLoss 
from Losses.AntLoss import AngularErrorLoss

from datasets.LSMIdataloader import IllumDataset, ToTensor, worker_init_fn ,RandomColor,RandomRotateFlip
from utils.util import calculate_angular_error,apply_wb


def compute_ae_metrics(errors_tensor):
    """计算 AE 的详细统计指标 (Mean, Median, Tri-mean, Best 25%, Worst 25%)"""
    errors = errors_tensor.cpu().numpy()
    errors_sorted = np.sort(errors)
    n = len(errors)

    if n == 0: return {}

    # Mean
    mean_err = np.mean(errors_sorted)
    # Median
    median_err = np.median(errors_sorted)
    # Tri-mean: (Q1 + 2*Q2 + Q3) / 4
    q1 = np.percentile(errors_sorted, 25)
    q3 = np.percentile(errors_sorted, 75)
    trimean_err = (q1 + 2 * median_err + q3) / 4
    # Best 25%
    num_best = max(int(n * 0.25), 1)
    best_25_mean = np.mean(errors_sorted[:num_best])
    # Worst 25%
    num_worst = max(int(n * 0.25), 1)
    worst_25_mean = np.mean(errors_sorted[-num_worst:])

    return {
        "mean": mean_err,
        "median": median_err,
        "trimean": trimean_err,
        "best_25": best_25_mean,
        "worst_25": worst_25_mean
    }

# ==========================================

# ==========================================
def count_model_params(model, unit="M"):
    total_params = sum(p.numel() for p in model.parameters())
    if unit == "M": return total_params / 1e6
    elif unit == "K": return total_params / 1e3
    else: return total_params

# ==========================================

# ==========================================
def process_files(directory):
    files_info = []
    if not os.path.exists(directory): return []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            try:
                
                parts = filename.replace('.pth', '').split('_')
                metric = float(parts[-2])
                epoch = int(parts[-1])
                files_info.append({'文件名': filename, 'Metric': metric, 'epoch': epoch})
            except:
                continue
    files_info.sort(key=lambda x: x['epoch'], reverse=True)
    return files_info


def eval_on_test_set(model, test_loader, criterion_L1, device, epoch, current_lr):
    model.eval()
    test_loss_accum = 0.0
    test_ae_all = []
    test_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Ep {epoch} Test", leave=False):
            input_tensor = batch['input'].to(device)
            target_illum = batch['gt'].to(device)
            mask = batch['mask'].to(device, non_blocking=True)
            
            preds_illum, _, _ = model(input_tensor)
            
            
            loss = criterion_L1(preds_illum, target_illum)
            test_loss_accum += loss.item()
            
            
            batch_aes = calculate_angular_error(preds_illum, target_illum, tensor_type="illumination", mask=mask)
            test_ae_all.append(batch_aes)
            
            test_steps += 1
    
    
    test_loss = test_loss_accum / test_steps
    test_ae_tensor = torch.cat(test_ae_all, dim=0)
    test_ae_stats = compute_ae_metrics(test_ae_tensor)
    
    
    test_log_msg = (f"Epoch {epoch} Test - Loss: {test_loss:.4f} | "
                    f"Mean AE: {test_ae_stats['mean']:.4f} | "
                    f"Median: {test_ae_stats['median']:.4f} | "
                    f"Trimean: {test_ae_stats['trimean']:.4f} | "
                    f"Best 25%: {test_ae_stats['best_25']:.4f} | "
                    f"Worst 25%: {test_ae_stats['worst_25']:.4f} | LR: {current_lr:.8f}")
    print(f"\n📊 {test_log_msg}\n")
    return test_loss, test_ae_stats

# ==========================================

# ==========================================
def train_and_evaluate(args):
    
    alpha, beta = 1, 3
    weight_dir = os.path.join(args.result_dir, f'alpha_{alpha}_beta_{beta}')
    os.makedirs(weight_dir, exist_ok=True)
    
    sub_dirs = ['best_loss_model', 'last_model', 'best_val_ae_model', 'best_test_ae_model', 'logs']
    for d in sub_dirs:
        os.makedirs(os.path.join(weight_dir, d), exist_ok=True)


    
    transform = transforms.Compose([RandomRotateFlip(split=args.split), ToTensor()])
    random_color_aug = None
    if args.illum_augmentation == 'yes':
        transform = transforms.Compose([RandomRotateFlip(split=args.split), ToTensor()])
    else:
        transform = transforms.Compose([ToTensor()])

    print(f"🚀 Loading Train Set ({args.split})...")
    traindataset = IllumDataset(
        root=args.data_root, split=args.split, illum_mode=args.illum_mode, 
        split_json_path=args.split_json_path, image_pool=args.image_pool,
        output_type="illum", transform=transform,illum_augmentation=random_color_aug
    )
    
    print(f"🚀 Loading Val Set (val)...")
    valdataset = IllumDataset(
        root=args.data_root, split='val', illum_mode=args.illum_mode, 
        split_json_path=args.split_json_path, image_pool=args.image_pool,
        output_type="illum", transform=transform
    )

    print(f"🚀 Loading Test Set (test)...")
    testdataset = IllumDataset(
        root=args.data_root, split='test', illum_mode=args.illum_mode, 
        split_json_path=args.split_json_path, image_pool=args.image_pool,
        output_type="illum", transform=transform
    )

    if len(traindataset) == 0:
        print("❌ 错误：训练集为空，请检查路径配置。")
        return

    train_loader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(valdataset, batch_size=1, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DUNF().to(device)
    print(f"模型参数量: {count_model_params(model):.2f} M")

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张 GPU 并行训练")
        model = nn.DataParallel(model)

    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_Color = ColorHistogramKLLoss().cuda()
    criterion_Angular = AngularErrorLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=20, cooldown=10, min_lr=1e-8
    )

    # ============================================================
    
    # ============================================================
    start_epoch = 1
    best_val_ae = float('inf') 
    best_test_ae = float('inf')

    
    save_lastmodel = os.path.join(weight_dir, 'last_model')
    save_best_val_dir = os.path.join(weight_dir, 'best_val_ae_model')
    save_best_test_dir = os.path.join(weight_dir, 'best_test_ae_model')

    if args.resume:
        
        last_files = process_files(save_lastmodel)
        if len(last_files) > 0:
            latest = last_files[0]
            model_path = os.path.join(save_lastmodel, latest['文件名'])
            print(f"🔄 恢复训练: {model_path}")
            
            checkpoint = torch.load(model_path)
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
            
            start_epoch = latest['epoch'] + 1

            
            
            val_files = process_files(save_best_val_dir)
            if len(val_files) > 0:
                
                best_val_ae = min([f['Metric'] for f in val_files])
                print(f"📈 已恢复历史最佳 Val AE: {best_val_ae:.4f}")
            
            
            test_files = process_files(save_best_test_dir)
            if len(test_files) > 0:
                best_test_ae = min([f['Metric'] for f in test_files])
                print(f"📈 已恢复历史最佳 Test AE: {best_test_ae:.4f}")

        else:
            print("⚠️ 未找到 Last Checkpoint，从头开始训练")

    
    print(f"🏁 开始训练 (Epoch {start_epoch} -> {args.num_epoch})")
    eps = 1e-8
    for epoch in range(start_epoch, args.num_epoch + 1):
        model.train()
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        train_metrics = {'loss': 0, 'p_loss': 0, 'q_loss': 0, 'ae': 0}
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch} Train")
        for batch in pbar:
            input_tensor = batch['input'].to(device, non_blocking=True)
            target_illum = batch['gt'].to(device, non_blocking=True)
            target_rgb = batch['gt_rgb'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            
            preds_illum, aux_loss, _ = model(input_tensor)

             
            loss_l1 = criterion_L1(preds_illum*mask, target_illum*mask)
            loss_ang = criterion_Angular(preds_illum, target_illum, mask)

            corrected_img = apply_wb(input_tensor, preds_illum, 'illumination')
            corrected_img = torch.clamp(corrected_img, 0.0, 1.0)
            
            loss_rgb_l1 = criterion_L1(corrected_img*mask, target_rgb*mask)
            loss_col = criterion_Color(corrected_img, target_rgb)

            total_loss = 0.3*loss_l1+0.7*loss_ang # + loss_col  #+ loss_rgb_l1 #+ 0.005*loss_ang# + 0*aux_loss 

            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            steps += 1
            train_metrics['loss'] += total_loss.item()
            train_metrics['p_loss'] += 0
            train_metrics['q_loss'] += 0
            
            with torch.no_grad():
                batch_ae = calculate_angular_error(preds_illum,target_illum,tensor_type="illumination",mask=mask)
                train_metrics['ae'] += batch_ae.mean().item()
            
            pbar.set_postfix(loss=total_loss.item(),LossL1 =loss_l1.item(),Loss_col=loss_col.item(),loss_rgb = loss_rgb_l1.item(),Loss_ang=loss_ang.item())#, ae=batch_ae.mean().item(),Loss_aux=aux_loss.item()

        
        for k in train_metrics: train_metrics[k] /= steps
        epoch_time = time.time() - epoch_start
        
        log_msg = f"Ep {epoch} Train Done. Loss: {train_metrics['loss']:.4f} | AE: {train_metrics['ae']:.4f}° | LR: {current_lr:.8f} | Time: {epoch_time:.1f}s,last_epoch: {scheduler.last_epoch}"
        print(log_msg) 

        
        last_path = os.path.join(save_lastmodel, f"model_last_{train_metrics['loss']:.4f}_{epoch}.pth")
        for f in os.listdir(save_lastmodel): os.remove(os.path.join(save_lastmodel, f))
        torch.save(model.state_dict(), last_path)

        
        if epoch % 2 == 0:
            model.eval()
            val_loss_accum = 0.0
            val_ae_all = [] 
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Ep {epoch} Val"):
                    input_tensor = batch['input'].to(device)
                    target_illum = batch['gt'].to(device)
                    mask = batch['mask'].to(device, non_blocking=True)
                    
                    preds_illum, _, _ = model(input_tensor)
                    
                    loss = criterion_L1(preds_illum, target_illum)
                    val_loss_accum += loss.item()
                    
                    batch_aes = calculate_angular_error(preds_illum, target_illum,tensor_type="illumination",mask=mask)
                    val_ae_all.append(batch_aes)
                    val_steps += 1
            
            val_loss = val_loss_accum / val_steps
            val_ae_tensor = torch.cat(val_ae_all, dim=0)
            ae_stats = compute_ae_metrics(val_ae_tensor)
            
            scheduler.step(ae_stats['mean'])
            
            val_log_msg = (f"Epoch {epoch} Val - Mean AE: {ae_stats['mean']:.4f} | Best Val: {best_val_ae:.4f}")
            print(f"\n📊 {val_log_msg}\n")

            
            if ae_stats['mean'] < best_val_ae:
                best_val_ae = ae_stats['mean']
                save_path = os.path.join(save_best_val_dir, f"best_val_ae_{best_val_ae:.4f}_{epoch}.pth")
                
                for f in os.listdir(save_best_val_dir): os.remove(os.path.join(save_best_val_dir, f))
                torch.save(model.state_dict(), save_path)
                print(f"🔥 New Best Val AE: {best_val_ae:.4f}° | 评估测试集...")
                
                
                test_loss, test_ae_stats = eval_on_test_set(model, test_loader, criterion_L1, device, epoch, current_lr)
                
                
                if test_ae_stats['mean'] < best_test_ae:
                    best_test_ae = test_ae_stats['mean']
                    test_save_path = os.path.join(save_best_test_dir, f"best_test_ae_{best_test_ae:.4f}_{epoch}.pth")
                    for f in os.listdir(save_best_test_dir): os.remove(os.path.join(save_best_test_dir, f))
                    torch.save(model.state_dict(), test_save_path)
                    print(f"🎉 New Best Test AE: {best_test_ae:.4f}°")

        
        if epoch % 10 == 0:
            print(f"\n🔍 Epoch {epoch} 强制评估测试集...")
            test_loss, test_ae_stats = eval_on_test_set(model, test_loader, criterion_L1, device, epoch, current_lr)
                            
            if test_ae_stats['mean'] < best_test_ae:
                best_test_ae = test_ae_stats['mean']
                test_save_path = os.path.join(weight_dir, 'best_test_ae_model', f"best_test_ae_{best_test_ae:.4f}_{epoch}.pth")
                test_save_dir = os.path.dirname(test_save_path)
                for f in os.listdir(test_save_dir): os.remove(os.path.join(test_save_dir, f))
                torch.save(model.state_dict(), test_save_path)
                print(f"🎉 New Best Test AE: {best_test_ae:.4f}° | 已保存最优测试集模型！\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="./LSMI/sony_256")
    parser.add_argument('--result_dir', type=str, default='./result/LSMI/AE_sony_mixed')
    parser.add_argument('--split_json_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--illum_mode', type=str, default='mixed')
    parser.add_argument('--image_pool', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_epoch', type=int, default=4001)
    parser.add_argument('--resume', action='store_true', default=True, help='是否从上次训练断点恢复')

    
    parser.add_argument('--gpu_id', type=str, default='0', help='指定使用的GPU ID，例如 0 或 0,1')

    parser.add_argument('--illum_augmentation', type=str, default='yes', help='是否启用光照增强')
    parser.add_argument('--sat_min', type=float, default=0.2)
    parser.add_argument('--sat_max', type=float, default=0.8)
    parser.add_argument('--val_min', type=float, default=1.0)
    parser.add_argument('--val_max', type=float, default=1.0)
    parser.add_argument('--hue_threshold', type=float, default=0.2)

    args = parser.parse_args()
    
    if args.split_json_path is None:
        args.split_json_path = os.path.join(args.data_root, "split.json")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    train_and_evaluate(args)



# illum_mode,image_pool,    实际加载的数据后缀   ,说明
# single,    [1]            ,_1                 ,标准单光源模式
# single,    "[1, 2, 3]"    ,_1                 ,因为 single 限制了只有 _1，即便池子里有 2 和 3 也不会加载
# multi,     "[2, 3]"       ,"_12, _13, _123"   ,标准多光源模式
# multi,     "[1, 2]",      _12                 ,multi 排除了 _1，image_pool 排除了 _123
# mixed,     "[1, 2, 3]"   ,"_1, _12, _13, _123",全量模式
# mixed,     "[1, 2]","    _1, _12, _13"        ,加载单光源和双光源，排除三光源



