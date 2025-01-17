import subprocess


def test_mujoco_py():
    """
    Test mujoco_py
    """
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --env-id Hopper-v2 --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/sac_continuous_action.py --env-id Hopper-v2 --batch-size 128 --total-timesteps 135",
        shell=True,
        check=True,
    )
