from pathlib import Path

import pytest

from plumeswift.xml_demo import read_simulation_demo, run_xml_demo


FIXTURES = Path(__file__).resolve().parents[2]/'external/QES-Public/scratch/xmlTesting'


def test_original_xml_examples_preserve_output_order_and_optional_fields():
    if not FIXTURES.is_dir():
        pytest.skip('Original scratch XML fixtures are not available')
    stdout, stderr, status = run_xml_demo(FIXTURES/'testPass.xml')
    assert stderr == '' and status == 0
    assert stdout.endswith('D\nP1\nthisIsP1A\n1\nP1\nthisIsP1B\n2\nP2\nthisIsP2A\nY\n7.12\nP2\nthisIsP2B\nZ\n8.13\n')
    assert run_xml_demo(FIXTURES/'testFail.xml') == ('', 'ERROR: Could not find root::X::D::P2::qVal\n', 0)
    assert run_xml_demo(FIXTURES/'testCantRead.xml') == ('', 'Error reading tree\n', 0)

    path = FIXTURES/'simulationParams/QU_simparams.xml'
    assert run_xml_demo(path, simulation=True) == ('', '', 0)
    config = read_simulation_demo(path)
    assert config['domain'] == [50, 50, 40] and config['cellSize'] == [1., 1., 1.]
    assert config['quicCDFFlag'] == 1
