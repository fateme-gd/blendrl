from nudge.renderer import Renderer

if __name__ == "__main__":
    renderer = Renderer(agent_path="out/runs/seaquest/deictic/24-05-07-15-59",
                        fps=15,
                        deterministic=False,
                        env_kwargs=dict(render_oc_overlay=True),
                        render_predicate_probs=True)
    renderer.run()
