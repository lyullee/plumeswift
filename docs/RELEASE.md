# PlumeSwift release procedure

1. Set the version in `python_qes/plumeswift/__init__.py` and update `CITATION.cff`.
2. Use GitHub Actions workflow **Build and publish Python package** with manual dispatch to build and check both distributions without publishing.
3. Run the installed wheel's synthetic smoke case and the component/numerics tests from outside the source tree. Inspect the wheel's file list and verify LICENSE, component notices and the HeatPot lookup table.
4. Generate the archival source from the release commit and record its SHA-256. Create a new Zenodo version when archiving corrected files. Record the DOI and retain the component-specific license notices in the source.
5. For PyPI, configure a narrowly scoped GitHub Actions Trusted Publisher for `lyullee/plumeswift`, workflow `release.yml`, environment `pypi`. Publish the matching GitHub release tagged `v<version>`. If GitHub does not start the release workflow, dispatch it from the default branch with `publish_tag` set to the existing public tag. The publishing job uploads the distributions validated in the build job.
6. Verify the published PyPI version, GitHub commit/tag and version DOI. Save the exact archive and checksum alongside the release notes.

The repository publishes Python sources, selected benchmark drivers, aggregated evidence and provenance. Keep the FFI report, extracted sensor records, private inputs and unreviewed research outputs outside release archives. The FFI command reads a user-supplied copy of the official PDF.
