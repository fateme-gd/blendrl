from nudge.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(\
        # agent_path="out/runs/kangaroo_actor_hybrid_blend_softmax_lr_0.00025_llr_0.00025_blr_0.00025_gamma_1.0_bentcoef_0.01_numenvs_20_steps_128_pretrained_False_joint_True_20",
                        # agent_path="out/runs/kangaroo_actor_hybrid_blend_softmax_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_20_steps_128_pretrained_False_joint_True_20",
                        agent_path="out/runs/kangaroo_actor_hybrid_blend_softmax_lr_0.00025_llr_0.0025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_20_steps_128_pretrained_False_joint_True_40",
                        env_name="kangaroo",
                        fps=30,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=True)
    renderer.run()
