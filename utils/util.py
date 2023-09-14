from datetime import datetime
import os

def make_expr_directory(save_path: str, expr_name: str) -> str:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    curdate = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_expr_list = os.listdir(save_path)
    
    if saved_expr_list == []:
        expr_num = 1
    else:
        expr_num = int(sorted(saved_expr_list, key=lambda x: int(x[:3]))[-1][:3]) + 1
    
    expr_save_path = os.path.join(save_path, f"{expr_num:03d}_{curdate}_{expr_name}")
    os.makedirs(expr_save_path, exist_ok=True)
    
    return expr_save_path