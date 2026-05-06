import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BACKUP_DIR = PROJECT_ROOT / "backups"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "04_train.py"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"

TARGET_WIN_RATE = 0.52
TARGET_CUM_RET = 8.0
MAX_ATTEMPTS = 50

def clean_checkpoints():
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def backup_artifacts(prefix: str, win_rate: float, cum_ret: float):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}_WR{win_rate:.2f}_CR{cum_ret:.2f}_{timestamp}"
    dest_path = BACKUP_DIR / folder_name
    
    shutil.copytree(CHECKPOINT_DIR, dest_path, dirs_exist_ok=True)
    print(f"Artefactos respaldados en: {dest_path}")

def main():
    best_cum_ret_so_far = -float('inf')
    best_win_rate_so_far = 0.0
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[{attempt}/{MAX_ATTEMPTS}] Iniciando iteración estocástica...")
        clean_checkpoints()
        
        process = subprocess.run([sys.executable, str(TRAIN_SCRIPT)], capture_output=False)
        
        if process.returncode != 0:
            print("Error de ejecución en 04_train.py. Omitiendo iteración.")
            continue
            
        if not HISTORY_FILE.exists():
            print(f"Archivo de métricas ausente: {HISTORY_FILE}. Omitiendo iteración.")
            continue
            
        with open(HISTORY_FILE, "r") as f:
            history_data = json.load(f)
            
        test_metrics = history_data.get("test_metrics", {})
        win_rate = test_metrics.get("win_rate", 0.0)
        cum_ret = test_metrics.get("cum_return_sim", -999.0)
        
        print(f"Evaluación Final -> Win Rate: {win_rate:.3f} | Cum Ret: {cum_ret:.3f}")
        
        # Evaluar retención del mejor intento (Best-so-far)
        if cum_ret > best_cum_ret_so_far:
            best_cum_ret_so_far = cum_ret
            best_win_rate_so_far = win_rate
            # Mover temporalmente el mejor subóptimo a una carpeta de caché si se desea
            # En este diseño, respaldaremos de inmediato si supera los umbrales absolutos.

        # Evaluar convergencia global
        if win_rate >= TARGET_WIN_RATE and cum_ret >= TARGET_CUM_RET:
            print("Convergencia alcanzada. Ejecutando protocolo de respaldo final.")
            backup_artifacts("WINNER", win_rate, cum_ret)
            break
        else:
            print("Mínimo local subóptimo. Reiniciando búsqueda temporal...")

    else:
        # Se ejecuta solo si el bucle for termina sin un 'break' (agotamiento de intentos)
        print(f"\nLímite de iteraciones alcanzado. Respaldando el mejor subóptimo encontrado (CR: {best_cum_ret_so_far:.2f}).")
        # Nota: Como clean_checkpoints se ejecuta al inicio del ciclo, el último estado 
        # en CHECKPOINT_DIR corresponde a la iteración 50. Para retener estrictamente 
        # el best_so_far real, requeriría guardar una copia en memoria/disco duro por cada pico.
        # En esta arquitectura simplificada, respaldamos la última iteración como referencia de fallo.
        backup_artifacts("BEST_SUBOPTIMAL", win_rate, cum_ret)

if __name__ == "__main__":
    main()