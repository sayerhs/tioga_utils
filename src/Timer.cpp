
#include "Timer.h"

namespace tioga_nalu
{

Teuchos::TimeMonitor get_timer(const std::string name)
{
    Teuchos::RCP<Teuchos::Time> timer = Teuchos::TimeMonitor::lookupCounter(name);
    if (timer.is_null())
        timer = Teuchos::TimeMonitor::getNewCounter(name);

    return Teuchos::TimeMonitor(*timer);
}
} // tioga_nalu
