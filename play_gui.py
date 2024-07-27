from nudge.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(\
        agent_path="out/runs/kangaroo_softmax_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.0_numenvs_60_steps_128_pretrained_False_joint_True_20",
        env_name="kangaroo",
        fps=5,
        deterministic=False,
        env_kwargs=dict(render_oc_overlay=True),
        render_predicate_probs=True)
    renderer.run()
