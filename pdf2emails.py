import os
import sys
import logging
import click
from google.cloud import vision
from google.cloud import storage
from poppler import load_from_file, PageRenderer
from io import BytesIO
from PIL import Image


root = logging.getLogger()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(message)s'))
root.addHandler(ch)
root.setLevel(logging.DEBUG)


log = logging.getLogger(__name__)


IMG_NAME = 'pdf2emails-current-image.png'


@click.command()
@click.option('--pdf', nargs=1, required=True, help="Path to pdf file")
@click.option('--gcloud-json-cred', nargs=1, required=True, help="Path to google cloud credential file in json format")
@click.option('--bucket-name', nargs=1, required=True, help="Name of a google storage bucket to upload each pdf image to before processing with google vision")
def main(pdf, gcloud_json_cred, bucket_name):
    """Extract email addresses from a pdf of scanned pages of email lists.

    \b
    Example:
    python pdf2emails.py --pdf list.pdf

    """

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcloud_json_cred

    svc = storage.Client.from_service_account_json(gcloud_json_cred)
    bucket = svc.get_bucket(bucket_name)

    gvc = vision.ImageAnnotatorClient()

    renderer = PageRenderer()
    pdf_document = load_from_file(pdf)

    emails = []

    for page_number in range(pdf_document.pages):
        page = pdf_document.create_page(page_number)
        image = renderer.render_page(page)

        # Convert pdf image to PIL image
        pil_image = Image.frombytes(
            "RGBA",
            (image.width, image.height),
            image.data,
            "raw",
            str(image.format),
        )

        # Export PIL image to a string
        sio = BytesIO()
        pil_image.save(sio, 'PNG', quality=100)
        contents = sio.getvalue()
        sio.close()

        # And upload PNG image to google storage bucket
        blob = bucket.blob(IMG_NAME)
        blob.upload_from_string(
            contents,
            content_type='image/png',
        )

        # Use vision api to extract all text in the image
        response = gvc.annotate_image({
            'image': {
                'source': {
                    'image_uri': 'gs://%s/%s' % (bucket_name, IMG_NAME),
                }
            },
            'features': [
                {
                    'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
                }
            ]
        })

        for t in response.text_annotations:
            # Assuming that if it has only one '@', it's an email
            # address... You'll have to tweak that if your pages don't only
            # contain email addresses
            if t.description.count('@') > 1:
                # That's the first annotation, gathering ALL text parsed in the
                # image Here we are assuming that all text blocks in the page
                # are email addresses. You'll have to tweak that if your pages
                # don't only contain email addresses
                for l in t.description.split('\n'):
                    if '@' in l:
                        l = l.strip().lower().replace(' ', '')
                        emails.append(l)

    # Remove duplicates:
    emails = sorted(list(set(emails)))

    log.info("-------------------------------------")
    for t in emails:
        log.info(t)
    log.info("-------------------------------------")

    log.info("Found %s emails" % len(emails))


main()
