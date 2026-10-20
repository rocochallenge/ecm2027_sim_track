class LocalAssetsOnly:
    def list_registry(self):
        return []

    def __getattr__(self, name):
        raise RuntimeError("This release loads bundled local assets only; remote asset acquisition is not supported.")

object_loader = LocalAssetsOnly()
