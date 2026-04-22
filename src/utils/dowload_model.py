import tempfile

import httpx


async def download_model(url: str) -> str:
    """Descarga modelo a archivo temporal y retorna la ruta"""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
