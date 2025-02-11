# About

A repository of the mostly python scripts I use to work with local and remote LLMs.

I typically work with local LLMs running with [Ollama](https://ollama.com/) and that operate on lower-power devices, such as my Raspberry Pi/ARM [Nomad](https://www.nomadproject.io/) Cluster or ARM-based Macs.

## Notes

Table of LLMs that I've worked with, and some of *my* quick notes mostly on their memory footprint.

| NAME                                    | ID            | SIZE     | Coment
----------------------------------------------------------------------------
| llama3.1:8b                             | 46e0c10c039e  | 4.9 GB	 | will run on pi5 8MB
| aratan/deepseek-1.5b-uj:latest          | e25ad2eb9ef7  | 1.0 GB   | will run on pi5 8MB, fast and kinda dumb
| llama2:latest                           | 78e26419b446  | 3.8 GB   | too large for pi5 8MB?
| dolphin-mistral.                        | 5dc8c5a2be65  | 4.1 GB   | will run on pi5 8MB
| everythinglm:latest                     | b005372bc34b  | 7.4 GB   | system memory (14.2 GiB)
| deepseek-coder:6.7b-instruct            | ce298d984115  | 3.8 GB	 | system memory (8.4 GiB)
| dolphin-phi								              | c5761fc77240  | 1.6 GB	 | will run on pi, kinda dumb, but also fast
| deepseek-r1:32b                         | 38056bbcbb2d  | 19 GB    | will run on Macs
