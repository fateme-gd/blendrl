from nudge.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(agent_path="out/runs/kangaroo",
                        env_name="kangaroo",
                        fps=20,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=True)
    renderer.run()
