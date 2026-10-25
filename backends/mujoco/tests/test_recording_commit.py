import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uni_teleop.recording.episode import EpisodeRecorder
from tests.test_teleop_control import append_frame


class RecordingCommitTest(unittest.TestCase):
    def saved_episode(self, directory):
        recorder = EpisodeRecorder(directory, "sweep", record_video=False)
        recorder.start()
        append_frame(recorder)
        path = recorder.finish()
        meta = recorder.session_directory / "meta"
        originals = {file.name: file.read_bytes() for file in meta.iterdir()}
        return recorder, path, originals

    def test_failed_or_interrupted_publication_restores_all_prior_manifests(self):
        for failure in (OSError, KeyboardInterrupt):
            with (
                self.subTest(failure=failure.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                recorder, saved, originals = self.saved_episode(directory)
                saved_bytes = saved.read_bytes()
                session = recorder.session_directory
                recorder.start()
                append_frame(recorder)
                replace = Path.replace

                def interrupt_info(
                    path, target, session=session, failure=failure, replace=replace
                ):
                    if path.name == "info.json" and path.parent.name.startswith(
                        ".metadata-"
                    ):
                        # The new episode manifest was already published before
                        # the last info marker failed. It must be rolled back.
                        episodes = (
                            (session / "meta/episodes.jsonl").read_text().splitlines()
                        )
                        self.assertEqual(len(episodes), 2)
                        raise failure("interrupted final manifest replacement")
                    return replace(path, target)

                with (
                    patch.object(Path, "replace", interrupt_info),
                    self.assertRaises(failure),
                ):
                    recorder.finish()
                self.assertTrue(recorder.active)
                self.assertTrue(recorder.write_failed)
                with self.assertRaisesRegex(RuntimeError, "abort"):
                    recorder.finish()
                self.assertEqual(len(recorder._episodes), 1)
                self.assertEqual(len(recorder._stats), 1)
                self.assertEqual(
                    {
                        file.name: file.read_bytes()
                        for file in (session / "meta").iterdir()
                    },
                    originals,
                )
                recorder.abort()
                self.assertEqual(saved.read_bytes(), saved_bytes)
                self.assertFalse((saved.parent / "episode_000001.parquet").exists())
                self.assertEqual(list(session.glob(".metadata-*")), [])

    def test_failed_rollback_keeps_recovery_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder, saved, originals = self.saved_episode(directory)
            session = recorder.session_directory
            recorder.start()
            append_frame(recorder)
            replace = Path.replace

            def fail_publish_and_restore(path, target):
                if path.name == "info.json" and path.parent.name.startswith(
                    ".metadata-"
                ):
                    raise OSError("publish failed")
                if path.parent.name == "restore" and path.name == "episodes.jsonl":
                    raise OSError("restore failed")
                return replace(path, target)

            with (
                patch.object(Path, "replace", fail_publish_and_restore),
                self.assertRaisesRegex(
                    RuntimeError, "Restore before using the dataset"
                ) as error,
            ):
                recorder.finish()
            self.assertTrue(recorder.write_failed)
            backups = list(session.glob(".metadata-*/previous"))
            self.assertEqual(len(backups), 1)
            self.assertIn(str(backups[0]), str(error.exception))
            self.assertEqual(
                {file.name: file.read_bytes() for file in backups[0].iterdir()},
                originals,
            )
            recorder.abort()
            self.assertTrue(saved.is_file())
            self.assertTrue(backups[0].is_dir())

    def test_interrupting_media_write_blocks_finish(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("uni_teleop.recording.episode._VideoWriter") as writer_class,
        ):
            writer = writer_class.return_value
            writer.add_frame.side_effect = KeyboardInterrupt("partial media frame")
            recorder = EpisodeRecorder(
                directory, "sweep", model=object(), data=object()
            )
            recorder.start()
            with self.assertRaises(KeyboardInterrupt):
                append_frame(recorder)
            self.assertEqual(recorder.frame_count, 0)
            self.assertTrue(recorder.write_failed)
            with self.assertRaisesRegex(RuntimeError, "abort"):
                recorder.finish()
            recorder.abort()
            writer.cancel.assert_called_once()
            self.assertFalse(recorder.write_failed)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_stats_failure_blocks_retry_and_preserves_saved_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder, saved, originals = self.saved_episode(directory)
            session = recorder.session_directory
            recorder.start()
            append_frame(recorder)
            with (
                patch.object(recorder, "_episode_stats", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                recorder.finish()
            self.assertTrue(recorder.write_failed)
            self.assertEqual(len(recorder._episodes), 1)
            recorder.abort()
            self.assertEqual(
                {file.name: file.read_bytes() for file in (session / "meta").iterdir()},
                originals,
            )
            self.assertTrue(saved.is_file())
            self.assertEqual(
                json.loads((session / "meta/info.json").read_text())["total_episodes"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
