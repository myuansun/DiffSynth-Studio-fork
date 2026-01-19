import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    # Initialize WandB if requested
    use_wandb = getattr(args, 'use_wandb', False) if args is not None else False
    wandb_project = getattr(args, 'wandb_project', 'diffsynth-training') if args is not None else 'diffsynth-training'
    wandb_run_name = getattr(args, 'wandb_run_name', None) if args is not None else None
    
    if use_wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "num_epochs": num_epochs,
                    "save_steps": save_steps,
                    "dataset_size": len(dataset),
                    "lora_rank": getattr(args, 'lora_rank', None),
                    "lora_target_modules": getattr(args, 'lora_target_modules', None),
                }
            )
            print(f"[WandB] Initialized project: {wandb_project}")
        except ImportError:
            print("[WandB] wandb not installed, skipping logging")
            use_wandb = False
    else:
        use_wandb = False  # Only log from main process
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    global_step = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                
                global_step += 1
                
                # Log to WandB
                if use_wandb:
                    import wandb
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch_id,
                        "global_step": global_step,
                    }, step=global_step)
                    
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            
        # Log epoch completion to WandB
        if use_wandb:
            import wandb
            wandb.log({"epoch_completed": epoch_id + 1}, step=global_step)
            
    model_logger.on_training_end(accelerator, model, save_steps)
    
    # Finish WandB run
    if use_wandb:
        import wandb
        wandb.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
