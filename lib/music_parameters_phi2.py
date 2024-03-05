from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import logging.config


def config():
    logging.config.fileConfig(
        fname='config.ini', disable_existing_loggers=False)

    # Get the logger specified in the file
    logger = logging.getLogger(__name__)

    return logger


logger = config()


def generate(labeled_scores):
    logger.info("Initialising tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True
    )

    logger.info("Success!")
    logger.info("Loading model Phi-2 from Hugging Face Transformers")

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        device_map="auto",
        trust_remote_code=True
    )

    logger.info("Success!")

    prompt = f"Generate musical parameters based on emotional data: {labeled_scores}. Instructions: Compose an expressive musical piece inspired by the provided emotional data. Tailor the musical parameters to embody the negativity, neutrality, and positivity values. Explore the intricate interplay of tempo, key, instrumentation, dynamics and so on to articulate the emotional nuances. For instance, a heightened negativity value might manifest in a slower tempo or a darker tonal palette. Unleash your creativity to forge a profound connection between emotions and music. Feel empowered to generate additional details or specify a desired musical genre to enhance the richness of the composition."

    with torch.no_grad():
        token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt")

        logger.info("Generating music parameters...")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3
            # top_k=50
            # top_p=0.9
        )

    output = tokenizer.decode(output_ids[0][token_ids.size(1):])
    print(output)

    logger.info("Music parameters successfully generated!")

    return output
