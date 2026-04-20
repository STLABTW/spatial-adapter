"""Unit tests for spatial_adapter.logger."""

import logging

from spatial_adapter.logger import setup_logger


class TestSetupLogger:
    def test_returns_logger(self):
        lg = setup_logger("test_basic")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "test_basic"

    def test_console_handler_attached(self):
        lg = setup_logger("test_console")
        handlers = [h for h in lg.handlers if isinstance(h, logging.StreamHandler)]
        assert len(handlers) >= 1

    def test_custom_level(self):
        lg = setup_logger("test_level", level=logging.DEBUG)
        assert lg.level == logging.DEBUG

    def test_custom_format(self):
        fmt = "%(message)s"
        lg = setup_logger("test_fmt", format_string=fmt)
        assert lg.handlers[0].formatter._fmt == fmt

    def test_file_handler(self, tmp_path):
        log_file = tmp_path / "test.log"
        lg = setup_logger("test_file", log_file=log_file)
        lg.info("hello")
        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert log_file.exists()
        assert "hello" in log_file.read_text()

    def test_file_creates_parent_dirs(self, tmp_path):
        log_file = tmp_path / "sub" / "dir" / "test.log"
        lg = setup_logger("test_mkdir", log_file=log_file)
        lg.info("nested")
        assert log_file.exists()

    def test_clears_previous_handlers(self):
        lg = setup_logger("test_clear")
        n1 = len(lg.handlers)
        lg = setup_logger("test_clear")
        n2 = len(lg.handlers)
        assert n1 == n2  # should not accumulate

    def test_no_file_handler_by_default(self):
        lg = setup_logger("test_no_file")
        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0
