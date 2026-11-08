"""Resolve explicit practice-only recording overrides before opening outputs."""


def configure_recording(args):
    if not getattr(args, 'no_recording', False):
        return True
    if args.review_dir is not None:
        raise ValueError('--no-recording cannot be combined with --review-dir')
    # This explicit override wins over recording enabled by a recipe.
    args.deferred_recording = False
    args.profile = False
    return False
