# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.backend.vlm.vlm_analyze import ModelSingleton


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-http-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=False,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
    num_workers: int = 8,  # 并发 worker 数量
):
    if backend.startswith("vlm-"):
        backend = backend[4:]

    MODEL_PATH = "/home/youwei/bzh/model/opendatalab/MinerU2.5-2509-1.2B"
    predictor = ModelSingleton().get_model(backend, MODEL_PATH, server_url)

    time_dict = {}

    # 改为并发解析
    def _worker(idx: int):
        pdf_file_name = pdf_file_names[idx]
        _pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes_list[idx], start_page_id, end_page_id)
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        start_time = time.time()
        middle_json, infer_result = vlm_doc_analyze(
            _pdf_bytes, image_writer=image_writer, predictor=predictor
        )
        elapsed = time.time() - start_time
        pdf_info = middle_json["pdf_info"]

        _process_output(
            pdf_info,
            _pdf_bytes,
            pdf_file_name,
            local_md_dir,
            local_image_dir,
            md_writer,
            f_draw_layout_bbox,
            f_draw_span_bbox,
            f_dump_orig_pdf,
            f_dump_md,
            f_dump_content_list,
            f_dump_middle_json,
            f_dump_model_output,
            f_make_md_mode,
            middle_json,
            infer_result,
            is_pipeline=False,
        )
        return pdf_file_name, elapsed

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, idx): idx for idx in range(len(pdf_bytes_list))}
        for future in as_completed(futures):
            pdf_name, elapsed = future.result()
            time_dict[pdf_name] = elapsed

    json.dump(time_dict, open(os.path.join(output_dir, f"{backend}_{time.time()}.json"), "w"), ensure_ascii=False, indent=4)


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=True
):
    """处理输出文件"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")


def doc_benchmark(
    path_list: list[Path],
    output_dir,
    lang="ch",
    backend="pipeline",
    method="auto",
    server_url=None,
    start_page_id=0,
    end_page_id=None,
    num_workers: int = 10,
):
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)

        output_dir = os.path.join(output_dir, f"{backend}")
        os.makedirs(output_dir, exist_ok=True)

        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            num_workers=num_workers,
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="imgs", help="输入文档目录，支持pdf或图片")
    parser.add_argument("--output_dir", type=str, default="output", help="结果输出目录")
    
    parser.add_argument("--method", type=str, default="vlm", help="解析方式，对应 parse_method")
    parser.add_argument("--start_page_id", type=int, default=0, help="起始页码(从0开始)")
    parser.add_argument("--end_page_id", type=int, default=None, help="结束页码(包含)，默认到文档结尾")
    parser.add_argument("--lang", type=str, default="ch", help="语言代码")

    # parser.add_argument("--backend", type=str, default="vlm-http-client")
    parser.add_argument("--backend", type=str, default="vllm-engine")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8087", help="当backend为http-client时的服务器地址")
    parser.add_argument("--num_workers", type=int, default=5, help="并发worker数量")

    args = parser.parse_args()

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = os.path.join(__dir__, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    pdf_suffixes = ["pdf"]
    image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg"]

    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if guess_suffix_by_path(doc_path) in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    doc_benchmark(
        path_list=doc_path_list,
        output_dir=output_dir,
        lang=args.lang,
        backend=args.backend,
        method=args.method,
        server_url=args.server_url,
        start_page_id=args.start_page_id,
        end_page_id=args.end_page_id,
        num_workers=args.num_workers,
    )
