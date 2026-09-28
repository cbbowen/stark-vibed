//! The brush the suite paints with, and every way it lays a mark: a gesture driven
//! sample by sample, a stroke replayed whole, a fill, a selection.

// The siblings of this module, and the colours and corpus beside them: every
// helper here is re-exported from `common`, so one glob reaches all of them.
use super::*;
use stark_engine::Engine;
use stark_engine::command::Tool;
use stark_engine::command::{DocCommand, Edit, GestureCommand, InputSample, ViewCommand};
use stark_engine::path::DEFAULT_TOLERANCE;
use stark_model::Srgb;
use stark_model::document::LayerId;
use stark_model::document::{
    BrushParams, BrushShape, FillOp, SelectionMode, SelectionOp, SelectionShape,
};
use stark_model::geom::Vec2;
use stark_model::{Gradient, GradientStop};

/// A horizontal and a vertical stroke crossing at the canvas origin — the pair the
/// layer, group, blend and merge suites lay one per layer, so where they cross is
/// where two layers meet and where they do not is where one shows alone.
pub const H_STROKE: &[Vec2] = &[Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)];

pub const V_STROKE: &[Vec2] = &[Vec2::new(0.0, -60.0), Vec2::new(0.0, 60.0)];

pub fn brush(color: [f32; 3], radius: f32) -> BrushParams {
    BrushParams {
        effect: stark_model::document::BrushEffect::painted(color),
        size: radius,
        // `drain` is quoted per *radius* (§6.2), and this helper's radius is its
        // caller's — so the old flat `0.0015` per canvas px is written as the product
        // that restores it at whatever tip is asked for. Every golden that comes
        // through here therefore renders the exact stroke it was blessed on, which is
        // the point: the unit change is a change of interpretation, and no test in
        // the suite was measuring the interpretation.
        drain: 0.0015 * radius,
        // Remapped with §6.6's family (2026-09-07): what `hardness` names is the
        // falloff width, and the new family's widest is 0.50 of the radius where the
        // old one's was 0.80 — so the same number is a sharper tip. Carried over by
        // its place in the dial's range (`1 − old_width/0.80`), which is what keeps
        // the suite exercising the tip it was written for.
        shape: BrushShape::Round { hardness: 0.8 },
        ..BrushParams::default()
    }
}

/// The handful of edits the suite makes to [`brush`]'s starting point, named.
///
/// `brush` is deliberately a brush somebody could paint with rather than a neutral
/// one — it loads, it dithers, its tip is soft — and so almost every test that measures
/// something takes one thing back off it. Fifty-odd sites wrote `b.drain = 0.0` to do
/// that, which says what it sets and not what it is for.
///
/// Consuming builders rather than `&mut self` setters, so a brush a test never mutates
/// again does not have to be `mut` to be adjusted once.
pub trait TestBrush {
    /// No one-way load (§6.2). `brush` quotes `drain` per radius, so a stroke of a few
    /// hundred px runs dry partway along and the far end of a band is bare — which is
    /// right for a brush and wrong for anything measuring the band.
    #[must_use]
    fn undrained(self) -> Self;

    /// No per-stamp colour dither (§6.2) — what any check comparing two renders of the
    /// *same* stroke texel by texel needs, since the dither is seeded per stroke.
    #[must_use]
    fn unjittered(self) -> Self;

    /// A round tip of `hardness`: 1.0 for an edge sharp enough that a copy of it would
    /// be obvious, 0.0 for one whose falloff covers the whole reach (§6.6).
    #[must_use]
    fn hard(self, hardness: f32) -> Self;

    /// The effect's overall rate, whichever effect is in force
    /// ([`BrushEffect::set_flow`](stark_model::document::BrushEffect::set_flow)).
    #[must_use]
    fn flow(self, flow: f32) -> Self;

    /// The same brush as a **wet** one with these axes (§6.2), at the neutral flow.
    ///
    /// Through `BrushEffect::wet_with` rather than `make_wet`, which would carry the
    /// paint effect's flow across instead of pinning it to 1: the axes are quoted at
    /// the neutral pass, so a test that states them means them.
    #[must_use]
    fn wet(self, dynamics: stark_model::document::BrushDynamics) -> Self;
}

impl TestBrush for BrushParams {
    fn undrained(self) -> Self {
        Self { drain: 0.0, ..self }
    }

    fn unjittered(self) -> Self {
        Self {
            jitter: 0.0,
            ..self
        }
    }

    fn hard(self, hardness: f32) -> Self {
        Self {
            shape: BrushShape::Round { hardness },
            ..self
        }
    }

    fn flow(mut self, flow: f32) -> Self {
        self.effect.set_flow(flow);
        self
    }

    fn wet(self, dynamics: stark_model::document::BrushDynamics) -> Self {
        // Spelled out because `BrushParams::wet` is the inherent getter and `wet` is
        // also this trait's method: same receiver, different arity.
        let color = BrushParams::paint(&self)
            .map(|p| p.color)
            .or_else(|| BrushParams::wet(&self).map(|w| w.color))
            .expect("a paint or wet brush to make wet");
        Self {
            effect: stark_model::document::BrushEffect::wet_with(color, dynamics),
            ..self
        }
    }
}

/// A gesture about to be made: which tool, and the two dials the press carries.
///
/// **The one place `GestureCommand::Start`'s fields are named.** They were spelled
/// out at every site that opened a gesture, which made a new field on the press a
/// fifty-file edit — and made the two dials invisible, since a site that wanted the
/// default wrote it out beside one that did not.
///
/// The defaults are the frontend's: [`DEFAULT_TOLERANCE`] and no rope. A test that
/// varies one says which, and says nothing about the other.
#[derive(Clone, Copy)]
#[must_use]
pub struct Gesture {
    tool: Tool,
    tolerance: f32,
    rope: f32,
}

impl Gesture {
    pub fn new(tool: Tool) -> Self {
        Self {
            tool,
            tolerance: DEFAULT_TOLERANCE,
            rope: 0.0,
        }
    }

    /// The input tolerance the path is fitted to (§6.2) — the frontend's
    /// smoothness dial.
    pub fn tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// The towed tip's string, in canvas px (§6.2): the mark follows this far
    /// behind the pointer rather than at it.
    pub fn rope(mut self, rope: f32) -> Self {
        self.rope = rope;
        self
    }

    /// Press at the first sample and carry the rest, leaving the gesture open.
    pub fn drive<'e>(
        self,
        engine: &'e mut Engine,
        samples: impl IntoIterator<Item = InputSample>,
    ) -> Live<'e> {
        let mut it = samples.into_iter();
        let first = it.next().expect("a gesture presses somewhere");
        self.press(engine, first);
        let mut live = Live { engine };
        for s in it {
            live.to(s);
        }
        live
    }

    /// Press at `sample` and hand the engine straight back — for a test that has
    /// its own work to do between the reports.
    pub fn press(self, engine: &mut Engine, sample: InputSample) {
        engine.process(GestureCommand::Start {
            tool: self.tool,
            sample,
            tolerance: self.tolerance,
            rope: self.rope,
        });
    }
}

/// A gesture in flight, holding the engine it is being made on.
pub struct Live<'e> {
    engine: &'e mut Engine,
}

impl Live<'_> {
    /// One more report.
    pub fn to(&mut self, sample: InputSample) -> &mut Self {
        self.engine.process(GestureCommand::To { sample });
        self
    }

    /// Lift the pen, committing the stroke.
    pub fn end(self) {
        self.engine.process(GestureCommand::End);
    }
}

/// Paint and commit a stroke through `points` with an explicit brush.
pub fn stroke_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::set_brush(b));
    Gesture::new(Tool::Brush)
        .drive(engine, points.iter().copied().map(InputSample::at))
        .end();
}

/// [`stroke_with`], each point pressed at its own pressure — for a claim about
/// what the pen does *along* one stroke, where every other stroke here is
/// pressed home throughout.
pub fn stroke_pressed(engine: &mut Engine, b: BrushParams, points: &[(Vec2, f32)]) {
    engine.process(ViewCommand::set_brush(b));
    Gesture::new(Tool::Brush)
        .drive(
            engine,
            points.iter().map(|&(pos, pressure)| InputSample {
                pos,
                pressure,
                ..InputSample::default()
            }),
        )
        .end();
}

/// Paint and commit a stroke through the given canvas points with `color`.
pub fn paint(engine: &mut Engine, color: [f32; 3], radius: f32, points: &[Vec2]) {
    stroke_with(engine, brush(color, radius), points);
}

/// The **drained black smear** (§6.2): a wide wet tip that runs dry 200 canvas px in
/// — `drain` 0.4 per radius on an 80 px tip — so everything laid past that arrived
/// through the reservoir rather than off the brush.
///
/// Black because `BrushParams::default()` always painted here, and the goldens of it
/// are blessed on a black smear. Four tests are about what its reservoir does, and
/// each carried the literal.
pub fn drained_smear() -> BrushParams {
    BrushParams {
        size: 80.0,
        shape: BrushShape::Round { hardness: 0.95 },
        drain: 0.4,
        effect: stark_model::document::BrushEffect::wet_with(
            [0.0; 3],
            stark_model::document::BrushDynamics {
                add: 1.0,
                lift: 0.95,
                deposit: 0.95,
                ..Default::default()
            },
        ),
        ..BrushParams::default()
    }
}

/// The five runs [`drained_smear`] is asked over: one visible stretch of stroke with
/// five different tails, collinear, so the geometry *in view* is identical in all
/// five and all that differs is where the flattener's bisection put the segment
/// boundaries.
///
/// `golden::golden_drained_brush_length_independent` blesses the five pictures and
/// `dynamics::a_carried_stroke_is_independent_of_how_the_path_was_cut` measures how
/// far apart they are — two claims about one scene, so the scene is stated once.
pub fn drained_runs() -> [[Vec2; 3]; 5] {
    [200.0f32, 300.0, 400.0, 500.0, 600.0].map(|x| {
        [
            Vec2::new(-400.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(x, 0.0),
        ]
    })
}

/// A compact red blob centred at `c` — two crossing strokes, so it has height
/// variation for the lighting to bite on and a seam would show in it.
pub fn blob(engine: &mut Engine, c: Vec2) {
    paint(
        engine,
        palette::RED,
        16.0,
        &[c + Vec2::new(-18.0, 0.0), c + Vec2::new(18.0, 0.0)],
    );
    paint(
        engine,
        palette::RED,
        10.0,
        &[c + Vec2::new(0.0, -14.0), c + Vec2::new(0.0, 14.0)],
    );
}

/// A straight leg along `y` from `x0` to `x1` at one `pressure` — the two reports a
/// hand actually makes of a straight run, which is the point wherever this is used:
/// the ceiling is interpolated across each segment (§6.2, `Paint::opacity_ramp`), so
/// a stroke does not have to be finely reported for its ceiling to be smooth.
pub fn leg(x0: f32, x1: f32, y: f32, pressure: f32) -> Vec<(Vec2, f32)> {
    vec![(Vec2::new(x0, y), pressure), (Vec2::new(x1, y), pressure)]
}

/// Replace the selection with the rect `min..max`, feathered by `feather` (§6.8).
pub fn select_rect(engine: &mut Engine, min: Vec2, max: Vec2, feather: f32) {
    engine.process(DocCommand::Select(SelectionOp::new(
        SelectionMode::Replace,
        SelectionShape::rect_from_corners(min, max),
        feather,
    )));
}

/// Commit a fill of the rect `min..max` on the active layer, through the direct
/// command path. `opacity` is how far it covers, which is what the panel's slider
/// sets.
///
/// **A test wanting a bitwise-uniform field asks for `feather` 0 and must fill
/// rather than stroke.** A feather-0 fill lays one parcel value on every interior
/// texel; a swept deposit's per-segment sums leave ±1-ulp f16 speckle along the
/// stroke's own length (§6.2), which is real texture, so a byte-exact claim laid
/// with a stroke is measuring the deposit rather than whatever it meant to.
pub fn fill_rect(
    engine: &mut Engine,
    min: Vec2,
    max: Vec2,
    color: [f32; 3],
    feather: f32,
    opacity: f32,
) {
    let layer = engine.observe().active_layer;
    engine.process(DocCommand::Set(Edit::Fill {
        layer,
        op: FillOp::new(
            SelectionShape::rect_from_corners(min, max),
            feather,
            Srgb::new(color),
            opacity,
        ),
    }));
}

/// [`fill_rect`] over the whole [`SIZE`] canvas at feather 0 — the bed a wet-loop or
/// a warp test lays before asking what its pass did to it.
pub fn fill_canvas(engine: &mut Engine, color: [f32; 3], opacity: f32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    fill_rect(engine, -half, half, color, 0.0, opacity);
}

/// Run `cmd` and hand back the layer it minted.
///
/// A command answers with nothing (§4), so the id has to be read back — and
/// `layers.last()` is not it: the top of the stack is somebody else's layer the
/// moment a document holds two (the trap the AddFilterButton fell into once). This
/// diffs the ids the projection held before the command against after, so the answer
/// is what the command added, wherever it landed.
pub fn added(engine: &mut Engine, cmd: impl Into<stark_engine::command::InputCommand>) -> LayerId {
    let before: Vec<LayerId> = engine.observe().layers.iter().map(|l| l.id).collect();
    engine.process(cmd);
    engine
        .observe()
        .layers
        .iter()
        .map(|l| l.id)
        .find(|id| !before.contains(id))
        .expect("the command added a layer")
}

/// A gradient of the given stops, `(position, sRGB)` (§22).
pub fn gradient(stops: &[(f32, [f32; 3])]) -> Gradient {
    Gradient::new(
        stops
            .iter()
            .map(|&(t, color)| GradientStop {
                t,
                color: Srgb::new(color),
            })
            .collect(),
    )
    .expect("a gradient of the given stops")
}

/// Commit a stroke through `points` as a **replay** rather than a gesture: one
/// commit, rendered whole at the fold.
///
/// That is the canonical render — what a file, a redo and a peer each produce —
/// where [`stroke_with`] lands the live preview's own tiles (`PreparedStroke`,
/// §6.2), which sit within [`SEAM_LEVELS`] of it. A test whose claim is about
/// replay fidelity draws the reference side with this, so the claim stays exact.
pub fn replay_with(engine: &mut Engine, b: BrushParams, points: &[Vec2]) {
    engine.process(ViewCommand::set_brush(b));
    let samples: Vec<InputSample> = points.iter().copied().map(InputSample::at).collect();
    engine.replay_stroke(&samples);
}
