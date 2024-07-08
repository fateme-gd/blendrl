from nudge.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(agent_path="out/runs/Seaquest-v4_actor_hybrid_blender_logic_lr_0.00025_numenvs_10_steps_1024_pretrained_True_joint_True_0",
                        fps=15,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=True)
    renderer.run()
