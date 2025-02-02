#[test]
fn trybuild_tests() {
    let t = trybuild::TestCases::new();

    t.pass("tests/ui/pass/*.rs");
}
