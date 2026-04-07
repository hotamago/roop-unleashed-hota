class ProcessEntry:
    def __init__(self, filename: str, start: int, end: int, fps: float, file_signature=None, display_name=None):
        self.filename = filename
        self.finalname = None
        self.startframe = start
        self.endframe = end
        self.fps = fps
        self.file_signature = file_signature
        self.display_name = display_name or filename.split("\\")[-1].split("/")[-1]