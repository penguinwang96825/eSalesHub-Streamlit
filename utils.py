import logging
import random
import numpy as np


logger = logging.getLogger(__name__)


def seed_everything(seed=914):
    logger.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)


LEVEL2INDEX = {
    'sales_call_lead': 0,
    'sales_call_qualified': 1,
    'sales_call_quote': 2,
    'sales_call_appointment': 3,
    'sales_call_sale': 4, 
    'customer_service_call_chase': 5,
    'customer_service_call_general': 6,
    'customer_service_call_cancellation': 7
}

INDEX2LEVEL = {v:k for k, v in LEVEL2INDEX.items()}