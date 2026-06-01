from __future__ import annotations

import logging
import httpx
from app.config import settings
from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping
from app.integrations.tally.xml_builder import build_voucher_xml

logger = logging.getLogger(__name__)


class TallyConnectionError(Exception):
    """Exception raised when connection to TallyPrime server fails."""
    pass


class TallyClient:
    """
    Client for interacting with the TallyPrime local XML server.
    """

    @classmethod
    async def post_voucher(
        cls,
        invoice: ExtractedInvoice,
        ledger: LedgerMapping,
        host: str = None,
        port: int = None
    ) -> bool:
        """
        Posts a purchase voucher XML to the configured TallyPrime local server.
        Raises TallyConnectionError if Tally is offline or refuses the connection.
        """
        xml_data = build_voucher_xml(invoice, ledger)
        tally_host = host or settings.TALLY_HOST
        tally_port = port or settings.TALLY_PORT
        url = f"http://{tally_host}:{tally_port}"
        
        logger.info("Attempting to post voucher for invoice %s to Tally at %s", invoice.invoice_number, url)
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = {"Content-Type": "text/xml; charset=utf-8"}
                response = await client.post(url, content=xml_data, headers=headers)
                
                if response.status_code == 200:
                    logger.info("Successfully posted voucher to TallyPrime: %s", response.text)
                    # Note: TallyPrime XML API always returns status 200, but the response body
                    # contains <LINEERROR> tags if the business logic failed.
                    if "<LINEERROR>" in response.text:
                        logger.warning("Tally returned success status but body has LINEERROR: %s", response.text)
                        return False
                    return True
                else:
                    logger.error("Tally server returned non-200 status code: %d", response.status_code)
                    return False
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPError) as e:
            logger.error("Connection to TallyPrime server failed at %s: %s", url, e)
            raise TallyConnectionError(f"Could not reach Tally server at {url}: {e}") from e
