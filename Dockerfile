FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt update --fix-missing

WORKDIR /project

COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv
RUN uv sync --frozen

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

ENV VIRTUAL_ENV=/project/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PYTHONPATH="/project"

COPY src/ src/

CMD ["python", "src/optimizer.py"]