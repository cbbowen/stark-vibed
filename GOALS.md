# Stark

This is a 2D painting application written in Rust with a strong focus on:
* **Beautiful, natural brush strokes** that can affect channels other than color, like depth and wetness. Color channels use the Oklab color space for perceptual blending. This enables users to produce breathtaking works that look like the oil paintings of the old masters.
* **Performant painting and compositing** that deeply leverage GPU acceleration for a highly responsive user experience. Photoshop users wish it felt this good!
* **Powerful, intuitive digitial tools** like infinite canvas and undo history provide polish that transcends traditional technical limitations to always avoid frustrating the user.

## Project Structure

### Backend

Frontend-agnostic asynchronous backend that accepts input commands and exposes the current state to the frontend. Efficiently leverages GPU hardware across platforms via WGPU, using shaders to optimally implement compositing and stroke rendering. Uses advanced tiled rendering to provide an infinite canvas for the user to explore and draw on.

#### Inputs

* Top-level WGPU resources `wgpu::Instance`, `wgpu::Adapter`, `wgpu::Device`, etc.
* A stream of `InputCommand`s. `InputComand` is an enum of all the possible stateful user interactions with the backend: `StartStroke { tool: ToolId, sample: InputSample }`, `AddLayer { above: Option<LayerId> }`, `Undo`, etc.

#### Outputs

* Provides a way to render the current canvas state to a surface (transformed to allow pan and zoom).
* Provides a way to efficiently save the full history of actions to recreate the file. This is the primary save format. This approach enables undo after loading a file and rendering of timelapses.

#### Testing

Uses golden image tests to verify all functionality works correctly and consistently. This is enabled by separating the backend from the frontend.

#### Notable Dependencies
* `wgpu` (29.0)
	* Source: https://crates.io/crates/wgpu
	* Documentation: https://docs.rs/wgpu/29.0.3/wgpu/
* `wesl`
  * Source: https://crates.io/crates/wesl
	* Documentation: https://wesl-lang.dev/docs/Getting-Started-Rust
* `history`
	* Source: https://github.com/cbbowen/history
  * Documentation: https://cbbowen.github.io/history/history/index.html
* `rpds`
  * Source: https://crates.io/crates/rpds
  * Documentation: https://docs.rs/rpds/latest/rpds/

### Dioxus Frontend

Implements an intuitive web frontend using the Dioxus framework. Long-term plans to support multi-user editing in a peer-to-peer model.

#### Notable Dependencies

* `dioxus`
  * Source: https://crates.io/crates/dioxus
	* Documentation: https://docs.rs/dioxus/0.7.9/dioxus/
* `iroh`
  * Source: https://crates.io/crates/iroh
  * Documentation: https://docs.rs/iroh/latest/iroh/
