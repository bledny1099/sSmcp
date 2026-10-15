#!/usr/bin/env python3
"""Root entrypoint for sSmcp (socialSimple MCP)."""

from ssmcp.server import main, mcp

__all__ = ["main", "mcp"]

if __name__ == "__main__":
    main()
