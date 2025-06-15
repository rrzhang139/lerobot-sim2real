#!/bin/bash

# === STEP 0: HELPER SETUP ===
SEED=42
EXP_NAME="ppo-SO100GraspCube-v1-rgb-${SEED}"
CONFIG_FILE="env_config_new.json"
CHECKPOINT_NAME="ckpt_005000.pt"

#########################################
# PART 1: LOCAL MACHINE — PHYSICAL + CAMERA SETUP
#########################################

echo ""
echo "🔧 STEP 1.1: PHYSICAL SETUP — Mount Robot and Camera"
echo "----------------------------------------------------"
echo "📸 Mount your phone camera on a stand and connect it to your Mac (e.g., via Continuity Camera or EpocCam)."
echo "📏 Position the camera at approx [x=0.7m, y=0.37m, z=0.28m] relative to the SO100 robot base."
echo "🧱 Place the SO100 robot on the table and fix its base in a known position."
echo ""
read -p "✅ Press Enter when your camera and robot are mounted..."

echo ""
echo "🎯 STEP 1.2: Align Real Camera to Simulated View"
echo "------------------------------------------------"
python lerobot_sim2real/scripts/camera_alignment.py \
    --env-id="SO100GraspCube-v1" \
    --env-kwargs-json-path=${CONFIG_FILE}

echo ""
echo "🎥 STEP 1.3: CAPTURE GREENSCREEN BACKGROUND"
echo "-------------------------------------------"
echo "📤 Now carefully remove the robot from the table without moving the camera."
echo "This allows capturing a clean background image for greenscreen training."
read -p "🧹 Press Enter when the robot is removed..."

python lerobot_sim2real/scripts/capture_background_image.py \
    --env-id="SO100GraspCube-v1" \
    --env-kwargs-json-path=${CONFIG_FILE} \
    --out=greenscreen.png

echo ""
echo "✅ greenscreen.png saved. Now put the robot back in exactly the same place."
read -p "📦 Press Enter when robot is remounted and you are ready to train..."

#########################################
# PART 2: SSH CLUSTER — SIMULATION TRAINING
#########################################

echo ""
echo "🚀 STEP 2: Run on SSH Cluster — Train Policy"
echo "--------------------------------------------"
echo "Copy the following files to the cluster:"
echo ""
echo "scp ${CONFIG_FILE} greenscreen.png <you>@<cluster>:~/project/"
echo ""
read -p "📁 Press Enter once you've copied them and SSH'ed into the cluster..."

echo ""
echo "🏋️ Now on the SSH cluster, run the following:"
echo "---------------------------------------------"
echo ""
echo "python lerobot_sim2real/scripts/train_ppo_rgb.py \\"
echo "  --env-id=\"SO100GraspCube-v1\" \\"
echo "  --env-kwargs-json-path=${CONFIG_FILE} \\"
echo "  --ppo.seed=${SEED} \\"
echo "  --ppo.num_envs=1024 --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=32 \\"
echo "  --ppo.total_timesteps=100000000 \\"
echo "  --ppo.gamma=0.9 \\"
echo "  --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \\"
echo "  --ppo.exp-name=${EXP_NAME} \\"
echo "  --ppo.track --ppo.wandb_project_name=\"SO100-ManiSkill\""
echo ""
read -p "⏳ Wait until training finishes, then copy the best checkpoint and config back to local. Press Enter when done..."

#########################################
# PART 3: LOCAL MACHINE — DEPLOYMENT
#########################################

echo ""
echo "📥 STEP 3.1: Copy Checkpoint from Cluster to Local"
echo "--------------------------------------------------"
echo "scp <you>@<cluster>:~/project/runs/${EXP_NAME}/${CHECKPOINT_NAME} ./ckpt.pt"
echo "scp <you>@<cluster>:~/project/${CONFIG_FILE} ./env_config.json"
read -p "📦 Press Enter when the checkpoint and config are copied..."

echo ""
echo "🤖 STEP 3.2: Run Policy on Real SO100 Robot"
echo "-------------------------------------------"
echo "Make sure your robot and camera are connected and in the same position as before!"
read -p "✅ Press Enter to begin real-world policy execution..."

python lerobot_sim2real/scripts/eval_ppo_rgb.py \
    --env-id="SO100GraspCube-v1" \
    --env-kwargs-json-path=env_config.json \
    --checkpoint=final.pt \
    --no-continuous-eval \
    --control-freq=15

echo ""
echo "🏁 DONE! The robot has attempted the task with the trained policy."
