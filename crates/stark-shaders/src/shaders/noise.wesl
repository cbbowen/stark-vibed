fn _residual_1(u: f32) -> f32 {
	return ((u % 1) + frexp(u).fract) % 1;
}

fn _residual_2(u: vec2f) -> vec2f {
	return ((u % 1) + frexp(u).fract) % 1;
}

fn _residual_3(u: vec3f) -> vec3f {
	return ((u % 1) + frexp(u).fract) % 1;
}

fn _residual_4(u: vec4f) -> vec4f {
	return ((u % 1) + frexp(u).fract) % 1;
}

fn _scramble(p: vec4f) -> vec4f {
	const A = mat4x4f(
		104.96660772261987, 209.92775135746336, 227.81098381022335, 191.98557068846299,
		57.325987507685454, 88.33804517392682, 178.3642227811882, 41.30717806259662,
		176.01071333127803, 21.697587570417028, 207.15814245034863, 109.18032468909763,
		151.2683520129725, 171.8508836704048, 229.57826981659093, 54.111286915583804);
	let q = _residual_4(p * A);

	const B = mat4x4f(
		64.87875919702984, 28.134093730384393, 88.53338096865468, 184.28035114124438,
		112.35536618005656, 213.94721322950258, 65.21131559120045, 42.59733122668854,
		181.08027623605918, 25.302285560772617, 165.390903262566, 56.55635144617699,
		4.0740064116854455, 94.55330511383423, 169.6853124522701, 192.11942399581815);
	return _residual_4(q * B);
}

fn _noise_from_1(p: f32) -> vec4f {
	const A = vec4(163.00661070913245, 232.53078009831387, 196.776672949935, 212.83679916854584);
	return _scramble(_residual_4(p * A));
}

fn _noise_from_2(p: vec2f) -> vec4f {
	const A = mat4x2f(
		227.67368285964335, 151.63976363738095, 131.55451762557533, 36.83654749564718,
		212.9502007340391, 209.767279536377, 38.48232284939459, 74.44599526853312);
	return _scramble(_residual_4(p * A));
}

fn _noise_from_3(p: vec3f) -> vec4f {
	const A = mat4x3f(
		182.2518097882874, 245.04465820272014, 7.799719080688533, 169.65183698681474,
		9.153900447168184, 225.10249280746072, 81.5678660684215, 208.1323577730782,
		108.6533116558777, 247.72288643152368, 40.08218678512583, 190.76919197249558);
	return _scramble(_residual_4(p * A));
}

fn _noise_from_4(p: vec4f) -> vec4f {
	const A = mat4x4f(
		58.98623200649698, 85.96554490741295, 94.83011288438985, 247.00766839968006,
		210.7165821415716, 195.1934366522749, 166.0838370481573, 82.49473513300447,
		218.76597291413975, 153.10046731611197, 233.36968354198115, 85.9383600787994,
		90.98725558932381, 136.581238130389, 198.7787371666237, 250.20004320644813);
	return _scramble(_residual_4(p * A));
}

fn _noise_to_1(p: vec4f) -> f32 {
	const A = vec4(175.27558437710604, 69.84856661780117, 139.571461612358, 224.04243377630957);
	return _residual_1(dot(p, A));
}

fn _noise_to_2(p: vec4f) -> vec2f {
	const A = mat2x4f(
		52.994966910107394, 157.40742255956772, 28.819945657438097, 186.6646225450882,
		148.43160225792525, 30.360998092923722, 254.69347347146612, 218.36769776413848);
	return _residual_2(p * A);
}

fn _noise_to_3(p: vec4f) -> vec3f {
	const A = mat3x4f(
		122.47217941304064, 63.150189715318305, 26.90181682698679, 91.28677044857156,
		73.60401475969772, 243.5979331807022, 229.603383283309, 111.44953911325939,
		96.7857754245671, 200.69558502553443, 223.08722569692054, 123.35553042794007);
	return _residual_3(p * A);
}

fn _noise_to_4(p: vec4f) -> vec4f {
	const A = mat4x4f(
		53.443038908372046, 179.390025078717, 190.10636219829252, 54.56199072229842,
		245.87447166262183, 62.22491032429514, 242.6867446918381, 151.48523216519325,
		184.08080491314797, 198.9471040460606, 30.96518051086278, 39.454340786722895,
		81.61217910556283, 55.45536381963532, 94.32533970095761, 92.49012961343655);
	return _residual_4(p * A);
}

// Public API
// TODO: Use seed to modulate PRNG.

fn noise_1_to_1(p: f32, seed: vec4f) -> f32 {
	return _noise_to_1(_noise_from_1(p));
}

fn noise_1_to_2(p: f32, seed: vec4f) -> vec2f {
	return _noise_to_2(_noise_from_1(p));
}

fn noise_1_to_3(p: f32, seed: vec4f) -> vec3f {
	return _noise_to_3(_noise_from_1(p));
}

fn noise_1_to_4(p: f32, seed: vec4f) -> vec4f {
	return _noise_to_4(_noise_from_1(p));
}

fn noise_2_to_1(p: vec2f, seed: vec4f) -> f32 {
	return _noise_to_1(_noise_from_2(p));
}

fn noise_2_to_2(p: vec2f, seed: vec4f) -> vec2f {
	return _noise_to_2(_noise_from_2(p));
}

fn noise_2_to_3(p: vec2f, seed: vec4f) -> vec3f {
	return _noise_to_3(_noise_from_2(p));
}

fn noise_2_to_4(p: vec2f, seed: vec4f) -> vec4f {
	return _noise_to_4(_noise_from_2(p));
}

fn noise_3_to_1(p: vec3f, seed: vec4f) -> f32 {
	return _noise_to_1(_noise_from_3(p));
}

fn noise_3_to_2(p: vec3f, seed: vec4f) -> vec2f {
	return _noise_to_2(_noise_from_3(p));
}

fn noise_3_to_3(p: vec3f, seed: vec4f) -> vec3f {
	return _noise_to_3(_noise_from_3(p));
}

fn noise_3_to_4(p: vec3f, seed: vec4f) -> vec4f {
	return _noise_to_4(_noise_from_3(p));
}

fn noise_4_to_1(p: vec4f, seed: vec4f) -> f32 {
	return _noise_to_1(_noise_from_4(p));
}

fn noise_4_to_2(p: vec4f, seed: vec4f) -> vec2f {
	return _noise_to_2(_noise_from_4(p));
}

fn noise_4_to_3(p: vec4f, seed: vec4f) -> vec3f {
	return _noise_to_3(_noise_from_4(p));
}

fn noise_4_to_4(p: vec4f, seed: vec4f) -> vec4f {
	return _noise_to_4(_noise_from_4(p));
}