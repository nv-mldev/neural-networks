from manim import *
import numpy as np

class PerceptronConvergenceGeometry(ThreeDScene):
    def construct(self):
        # Set up the 3D scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # Title
        title = Text("Perceptron Convergence: Geometric Interpretation", font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait()

        # Add axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 5, 1],
            x_length=6,
            y_length=6,
            z_length=5,
        )

        # Optimal weight vector w* (pointing upward and slightly forward)
        w_star = Arrow3D(
            start=ORIGIN,
            end=[0.5, 0.5, 4],
            color=GREEN,
            thickness=0.02,
        )
        w_star_label = MathTex(r"\mathbf{w}^*", color=GREEN, font_size=48)
        w_star_label.next_to([0.5, 0.5, 4], RIGHT)
        self.add_fixed_in_frame_mobjects(w_star_label)

        self.play(Create(axes), Create(w_star))
        self.play(Write(w_star_label))
        self.wait()

        # Create cone and sphere for multiple iterations
        num_iterations = 5

        for k in range(1, num_iterations + 1):
            # Clear previous iteration label if exists
            if k > 1:
                self.play(FadeOut(iteration_label), run_time=0.3)

            # Iteration label
            iteration_label = Text(f"After k = {k} mistakes", font_size=32, color=YELLOW)
            iteration_label.to_edge(DOWN)
            self.add_fixed_in_frame_mobjects(iteration_label)
            self.play(Write(iteration_label))

            # Cone angle decreases with iterations (narrowing cone)
            # Angle decreases as alignment improves
            base_angle = 60  # degrees
            cone_angle = base_angle / np.sqrt(k + 1)  # Narrows as k increases
            cone_height = 4

            # Create cone
            cone = Cone(
                base_radius=cone_height * np.tan(cone_angle * DEGREES),
                height=cone_height,
                direction=[0.5, 0.5, 4] / np.linalg.norm([0.5, 0.5, 4]),
                resolution=(20, 20),
            )
            cone.set_color(BLUE)
            cone.set_opacity(0.3)
            cone.move_to([0, 0, cone_height/2])

            # Sphere radius grows with sqrt(k)
            sphere_radius = 0.8 * np.sqrt(k)
            sphere = Sphere(radius=sphere_radius, resolution=(20, 20))
            sphere.set_color(RED)
            sphere.set_opacity(0.2)
            sphere.move_to(ORIGIN)

            # Current weight vector w_k (inside both cone and sphere)
            # Position it to respect both constraints
            max_radius = min(sphere_radius * 0.8, cone_height * 0.8)
            angle_within_cone = cone_angle * 0.7 * DEGREES  # Stay well within cone

            w_k_length = max_radius
            # Direction: within the cone around w*
            w_k_end = [
                0.5 + w_k_length * np.sin(angle_within_cone),
                0.5 + w_k_length * np.sin(angle_within_cone),
                w_k_length * np.cos(angle_within_cone)
            ]

            w_k = Arrow3D(
                start=ORIGIN,
                end=w_k_end,
                color=YELLOW,
                thickness=0.015,
            )

            w_k_label = MathTex(r"\mathbf{w}_" + str(k), color=YELLOW, font_size=40)
            w_k_label.next_to(w_k_end, RIGHT)
            self.add_fixed_in_frame_mobjects(w_k_label)

            # Animate the creation
            if k == 1:
                self.play(
                    Create(cone),
                    Create(sphere),
                    Create(w_k),
                    Write(w_k_label),
                    run_time=2
                )
            else:
                # Fade out previous iteration
                self.play(
                    FadeOut(prev_cone),
                    FadeOut(prev_sphere),
                    FadeOut(prev_w_k),
                    FadeOut(prev_w_k_label),
                    run_time=0.5
                )
                # Show new iteration
                self.play(
                    Create(cone),
                    Create(sphere),
                    Create(w_k),
                    Write(w_k_label),
                    run_time=1.5
                )

            # Labels for cone and sphere
            if k == 1:
                cone_label = Text("Cone of Progress\n(narrows with k)",
                                 font_size=24, color=BLUE)
                cone_label.to_corner(UL)
                cone_label.shift(DOWN * 1.5)

                sphere_label = Text("Sphere of Possibility\n(grows as √k)",
                                   font_size=24, color=RED)
                sphere_label.to_corner(UR)
                sphere_label.shift(DOWN * 1.5)

                self.add_fixed_in_frame_mobjects(cone_label, sphere_label)
                self.play(Write(cone_label), Write(sphere_label))

            self.wait(1.5)

            # Store for next iteration
            prev_cone = cone
            prev_sphere = sphere
            prev_w_k = w_k
            prev_w_k_label = w_k_label

        # Final message
        self.play(FadeOut(iteration_label))
        final_text = Text(
            "The cone narrows faster than the sphere grows!\nConvergence is guaranteed.",
            font_size=32,
            color=YELLOW
        )
        final_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final_text)
        self.play(Write(final_text))

        # Rotate camera for better view
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        self.wait(2)


class PerceptronConvergence2D(Scene):
    """2D version showing the bounds more clearly"""
    def construct(self):
        # Title
        title = Text("Perceptron Convergence: The Two Bounds", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_tip": True},
        )

        # Labels
        x_label = MathTex("k \\ \\text{(number of mistakes)}", font_size=32)
        x_label.next_to(axes.x_axis, DOWN)

        y_label = Text("Growth", font_size=32)
        y_label.next_to(axes.y_axis, LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait()

        # Lower bound: alignment grows linearly
        lower_bound = axes.plot(
            lambda x: 0.5 * x,
            x_range=[0, 10],
            color=BLUE
        )
        lower_label = MathTex(
            r"\mathbf{w}_k \cdot \mathbf{w}^* \geq k\gamma \\ \text{(Linear growth)}",
            font_size=32,
            color=BLUE
        )
        lower_label.next_to(axes.c2p(8, 4), UP)

        self.play(Create(lower_bound), Write(lower_label))
        self.wait()

        # Upper bound: squared norm grows linearly
        upper_bound = axes.plot(
            lambda x: 0.5 * x,
            x_range=[0, 10],
            color=RED
        )
        upper_label = MathTex(
            r"||\mathbf{w}_k||^2 \leq kR^2 \\ \text{(Linear growth)}",
            font_size=32,
            color=RED
        )
        upper_label.next_to(axes.c2p(8, 3), DOWN)

        self.play(Create(upper_bound), Write(upper_label))
        self.wait()

        # Show Cauchy-Schwarz brings them together
        cauchy_text = MathTex(
            r"\text{Cauchy-Schwarz: } (\mathbf{w}_k \cdot \mathbf{w}^*)^2 \leq ||\mathbf{w}_k||^2 ||\mathbf{w}^*||^2",
            font_size=28,
            color=YELLOW
        )
        cauchy_text.to_edge(DOWN)
        self.play(Write(cauchy_text))
        self.wait()

        # Show the bound
        bound_arrow = Arrow(
            start=axes.c2p(6, 0),
            end=axes.c2p(6, 0.5),
            color=GREEN,
            buff=0
        )
        bound_text = MathTex(
            r"k \leq \frac{R^2}{\gamma^2}",
            font_size=36,
            color=GREEN
        )
        bound_text.next_to(bound_arrow, LEFT)

        self.play(
            Create(bound_arrow),
            Write(bound_text)
        )
        self.wait()

        # Final conclusion
        conclusion = Text(
            "Algorithm must terminate in finite steps!",
            font_size=36,
            color=GREEN
        )
        conclusion.move_to(cauchy_text.get_center())

        self.play(
            FadeOut(cauchy_text),
            Write(conclusion)
        )
        self.wait(2)


class PerceptronConvergenceCombined(Scene):
    """Combined animation showing both 2D and 3D views"""
    def construct(self):
        # First show the 2D explanation
        two_d_scene = PerceptronConvergence2D()
        two_d_scene.construct()

        # Transition text
        self.clear()
        transition = Text("Now let's see the 3D geometric interpretation...", font_size=36)
        self.play(Write(transition))
        self.wait(2)
        self.play(FadeOut(transition))

        # Then show the 3D visualization
        # Note: This won't work directly because ThreeDScene and Scene are different
        # We need to render them separately or use a different approach


if __name__ == "__main__":
    # To render individual videos:
    # manim -pqh perceptron_convergence_geometry.py PerceptronConvergenceGeometry
    # manim -pqh perceptron_convergence_geometry.py PerceptronConvergence2D

    # To render just the 3D geometric interpretation (recommended):
    # manim -pqh perceptron_convergence_geometry.py PerceptronConvergenceGeometry
    pass
