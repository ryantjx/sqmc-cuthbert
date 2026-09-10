"""Refresh credentials and execute through Colab without pruning the session.

Run with the CLI's Python interpreter. The installed CLI's keep-alive maintains
the VM but does not update its cached, expiring runtime proxy credentials.
Its exec command prunes session state on 401/404; this adapter uses the same
transport directly and leaves the VM, kernel and keep-alive intact on error.
"""

import argparse
import json
from pathlib import Path


def refresh(state, name, endpoint, *, session_factory=None, keep_alive=None):
    session = state.store.get(name)
    if session is not None and session.endpoint != endpoint:
        raise RuntimeError("Owned Colab session endpoint changed")
    assignment = next((item for item in state.client.list_assignments()
                       if item.endpoint == endpoint), None)
    if assignment is None:
        raise RuntimeError("Owned Colab runtime is no longer assigned")
    proxy = assignment.runtime_proxy_info
    restored = session is None
    if restored:
        if session_factory is None:
            from colab_cli.state import SessionState
            session_factory = SessionState

        # The server, not the disposable CLI cache, decides whether the saved
        # endpoint still exists. Never allocate a replacement VM or steal an alias.
        if any(item.endpoint == endpoint for item in state.store.list().values()):
            raise RuntimeError("Saved endpoint is registered under a different session name")
        session = session_factory(name=name, endpoint=endpoint, token=proxy.token, url=proxy.url,
                               variant=assignment.variant.name, accelerator=assignment.accelerator.value)
    # Update credentials only for the existing VM; keep its kernel and session
    # identity so refreshing cannot provision or substitute a different run.
    session.token, session.url = proxy.token, proxy.url
    state.store.add(session)
    if restored or getattr(session, "keep_alive_pid", False) is None:
        if keep_alive is None:
            from colab_cli.commands.session import spawn_keep_alive
            keep_alive = spawn_keep_alive

        # CLI pruning also killed keep-alive. Restore it only after confirming
        # that this exact VM is still assigned; training is never relaunched.
        session.keep_alive_pid = keep_alive(endpoint, name,
                                                  auth_provider=state.auth_provider,
                                                  config_path=state.config_path)
        state.store.add(session)
    return {"expires_in_seconds": proxy.token_expires_in_seconds}


def execute(state, name, endpoint, code, timeout, runtime_factory=None):
    """Use the CLI transport without its destructive `exec` error handler."""
    if runtime_factory is None:
        from colab_cli.runtime import ColabRuntime
        runtime_factory = ColabRuntime
    refresh(state, name, endpoint)
    session = state.store.get(name)

    def remember(field, value):
        current = state.store.get(name)
        if current is None or current.endpoint != endpoint:
            raise RuntimeError("Owned session changed while connecting")
        setattr(current, field, value)
        state.store.add(current)

    runtime = runtime_factory(session.url, session.token,
                              kernel_id=session.kernel_id, session_id=session.session_id,
                              on_kernel_started=lambda value: remember("kernel_id", value),
                              on_session_started=lambda value: remember("session_id", value))
    try:
        outputs = runtime.execute_code(code, timeout=timeout)
        for output in outputs:
            if output.get("output_type") == "stream":
                print(output.get("text", ""), end="", flush=True)
            elif output.get("output_type") == "error":
                raise RuntimeError(f"{output.get('ename', 'RemoteError')}: {output.get('evalue', '')}")
    finally:
        # Close only our connection, including on 401/404 or timeout. Do not
        # prune the session, kill keep-alive, or shut down the remote kernel.
        runtime.stop(shutdown_kernel=False)


if __name__ == "__main__":
    from colab_cli.common import state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("endpoint")
    parser.add_argument("--exec-file", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.exec_file:
        execute(state, args.session, args.endpoint, args.exec_file.read_text(), args.timeout)
    else:
        print(json.dumps(refresh(state, args.session, args.endpoint)))
